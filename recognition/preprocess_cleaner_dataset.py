from PyQt5.QtWidgets import QApplication
app = QApplication([])

if __name__ == "__main__":
    import os
    import threading
    import queue

    import cv2
    from tqdm import tqdm
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader

    import config
    from .train_dataset import dataset, lookup_table
    from utils import ensure_dir, denormalize


    from torch.backends import cudnn
    cudnn.benchmark = True

    from config import device
    save_dir = config.recognition_dataset_cache
    ensure_dir(save_dir)

    from cleaner.residual_gan import Generator

    # load weights
    weight_path = config.cleaner_weights_path
    weight_name = "e_90"

    ckpt = torch.load(os.path.join(weight_path, f"checkpoint_{weight_name}.weight"))

    generator = Generator().to(device)
    generator.load_state_dict(ckpt["generator"])

    generator.eval()

    lookup_table_reverse = {key: value for (value, key) in lookup_table.items() }
    character_amount = {}

    def save_img(img, character_id):

        character = lookup_table_reverse[character_id]
        save_path = os.path.join(save_dir, character)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if character not in character_amount:
            character_amount[character] = 0
        
        character_amount[character] += 1
            
        resized = TF.resize(img, (64, 64))
        
        img_np = (resized * 255).permute(1, 2, 0).numpy()

        cv2.imwrite(os.path.join(save_path, f"{character_amount[character]}.png"), img_np)



    def save_batch(imgs, character_ids):
        num_imgs = imgs.size(0)
        for i in range(num_imgs):
            save_img(imgs[i], character_ids[i].item())
            

    loader = DataLoader(dataset, batch_size=1200, num_workers=4, shuffle=False)


    threads = []


    q = queue.Queue(maxsize=10)

    def save_worker(worker_id: int):
        print(f"Worker #{worker_id} started.")
        while True:
            try:
                imgs, character_ids = q.get(timeout=30)
                save_batch(imgs, character_ids)
            except queue.Empty:
                break
        print(f"Worker #{worker_id} destroyed.")

        

    num_save_workers = 5

    threads = []
    for id in range(num_save_workers):
        t = threading.Thread(target=save_worker, args=(id, ))
        t.start()
        threads.append(t)


    with torch.no_grad():
        for imgs, character_ids in tqdm(loader):
            with torch.cuda.amp.autocast(): 
                cleaned = generator(imgs.to(device))

            cleaned = denormalize(cleaned)

            #thread = threading.Thread(target=save_batch, args=(cleaned.float().cpu(), character_ids))
            #thread.start()
            #threads.append(thread)
            print("putting to queue...")
            q.put((cleaned.float().cpu(), character_ids))
            print("done.")

    for thread in threads:
        thread.join()
