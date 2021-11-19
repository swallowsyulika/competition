import re
pattern = re.compile("([A-z0-9]+)_([0-9]+)_((tb)|(lr))_([0-9]+).jpg")

def parse_filename(filename: str):
    m = pattern.match(filename)    
    img_name = m.group(1)
    container_id = m.group(2)
    orientation = m.group(3)
    character_id = m.group(6)

    return img_name, container_id, orientation, character_id

if __name__ == "__main__":
    print(parse_filename("img_6006_1_lr_1.jpg"))
    print(parse_filename("img_6009_5_tb_3.jpg"))