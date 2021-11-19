from datasets.font_dataset_qt import FontDatasetQt
import os

from albumentations.augmentations.geometric.transforms import Perspective
from torchvision.transforms.functional import scale
from torchvision.transforms.transforms import RandomApply
from utils.album_transform_adapter import TransformAdapter
from torchvision import transforms
from datasets import *
from custom_transforms import *
import config
from utils import get_character_list, filter_character_list_advanced, TransformAdapter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# paths
texture_dir = config.project_textures_path
fonts_path = config.project_fonts_path
character_list_file = config.cleaner_character_list


# load characters set
characters = get_character_list(character_list_file)

font_files = [
    "NotoSansCJK-Medium.ttc",
    "NotoSerifCJK-Medium.ttc",
    "TW-Kai-98_1.ttf",
    "jf-openhuninn-1.1.ttf",
    "setofont.ttf",
    "JasonWriting1.ttf",
    "JasonWriting2.ttf",
    "TaipeiSansTCBeta-Bold.ttf",
    "TaipeiSansTCBeta-Light.ttf",
    "TaipeiSansTCBeta-Regular.ttf",
    "aoyagireisyosimo_ttf_2_01.ttf",
    "KouzanBrushFontSousyo.ttf",
    "GenSenRounded-B.ttc"
]

font_files = [os.path.join(fonts_path, x) for x in font_files]

bg_generator = BackgroundGenerator(
    texture_dir,
    zones_range=(10, 30))

combined_datasets = []

font_size = 96
out_size = 128

# random_line = RandomLine(font_size, (0, 3), thickness=2, num_samples=500)

post_process = transforms.Compose([
    transforms.ToTensor(),
    # random_line,
    transforms.RandomApply([
        RandomStackBackground(bg_generator),
    ], p=0.5),
    TransformAdapter(A.Compose([
        A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ToTensorV2()
    ]), input_type='tensor'),
    RandomDropResolution((18, 128)),
    transforms.Resize((out_size, out_size)),
    transforms.Normalize(mean=0.5, std=0.5),

])

spatial_transform = A.Compose([
    A.Perspective(scale=(0.02, 0.1), pad_val=1),
    A.Affine(
        scale=(1, 1.1),
        rotate=(-10, 10), cval=1),
    ToTensorV2(),

], additional_targets=({
    "image0": 'image'
}))

for font_file in font_files:
    filtered_characters = filter_character_list_advanced(font_file, characters)
    dirty_datasets = []

    # background + random text color (Qt)
    dirty_datasets.append(FontDatasetQt(filtered_characters, font_file,
                                          font_size=font_size,
                                          img_size=font_size,
                                          bg_generator=bg_generator,
                                          random_character_color=True,
                                          transform=post_process))

    # random background + pseudo 3D text with random color (PIL)
    dirty_datasets.append(Pseudo3DFontDatasetQt(filtered_characters, font_file,
                                                  font_size=font_size,
                                                  img_size=font_size,
                                                  bg_generator=bg_generator,
                                                  transform=post_process))

    # random background + text with border (Qt)
    dirty_datasets.append(BorderedFontDataset(filtered_characters, font_file,
                                                 font_size=font_size,
                                                 img_size=font_size,
                                                 bg_generator=bg_generator,
                                                 transform=post_process))

    # random background + hollow text (Qt)
    dirty_datasets.append(HollowFontDataset(filtered_characters, font_file,
                                               font_size=font_size,
                                               img_size=font_size,
                                               bg_generator=bg_generator,
                                               border_size=(2, 4),
                                               transform=post_process))

    # random background + text border + text texture (Qt)
    dirty_datasets.append(TextureFontDataset(filtered_characters, font_file,
                                                font_size=font_size,
                                                img_size=font_size,
                                                bg_generator=bg_generator,
                                                border_size=(8, 16),
                                                transform=post_process))

    # random background + double text border + text texture + shadow (Qt)
    dirty_datasets.append(TextureDBorderFontDataset(filtered_characters, font_file,
                                                       font_size=font_size,
                                                       img_size=font_size,
                                                       bg_generator=bg_generator,
                                                       border_size=(8, 8),
                                                       out_offset=(8, 16),
                                                       shadow_offset_x=(-4, 4),
                                                       shadow_offset_y=(-4, 4),
                                                       transform=post_process))
    # white background + black text (Qt)
    dirty_datasets.append(FontDatasetQt(filtered_characters, font_file,
                                          font_size=font_size,
                                          img_size=font_size,
                                          transform=post_process))

    clean_ds = FontDatasetQt(
        filtered_characters,
        font_file,
        font_size=out_size,
        img_size=out_size,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(std=0.5, mean=0.5),
        ]))


    combined_datasets.append(CombinedDataset(
        dirty_datasets, clean_ds, transform=spatial_transform))

    # smaller fonts
    # soft shadow with white border (often fail case)
    sm_dirty_datasets = []
    sm_dirty_datasets.append(ShadowBorderDataset(filtered_characters, font_file,
                                                 font_size=out_size - 32,
                                                 img_size=out_size,
                                                 bg_generator=bg_generator,
                                                 transform=post_process))
    sm_clean_ds = FontDatasetQt(filtered_characters, font_file,
                              font_size=out_size - 32,
                              img_size=out_size,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(std=0.5, mean=0.5),
                              ]))

    combined_datasets.append(CombinedDataset(
        sm_dirty_datasets, sm_clean_ds, transform=spatial_transform))


dataset = ConcatDataset(*combined_datasets)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    clean, dirty = dataset[10]

    plt.imshow(clean[0])
    plt.show()

    plt.imshow(dirty[0])
    plt.show()
