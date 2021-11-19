import json
from torchvision import transforms
from datasets import *
from custom_transforms import *
from utils import filter_character_list

with open("characters.json", 'r') as f:
    characters = json.loads(f.read())

font_files = [
    # "/usr/share/fonts/noto-cjk/NotoSansCJK-Medium.ttc",
    # "/usr/share/fonts/noto-cjk/NotoSerifCJK-Medium.ttc",
    # "/home/tingyu/Open_Data/Fonts/TW-Kai-98_1.ttf",
    # # "/home/tingyu/Open_Data/Fonts/TW-Sung-98_1.ttf"
    # "/home/tingyu/jf-openhuninn-1.1/jf-openhuninn-1.1.ttf",
    # "/home/tingyu/setofont/setofont.ttf"

    "fonts/NotoSansCJK-Medium.ttc",
    "fonts/NotoSerifCJK-Medium.ttc",
    "fonts/TW-Kai-98_1.ttf",
    "fonts/jf-openhuninn-1.1.ttf",
    "fonts/setofont.ttf",
    "fonts/JasonWriting1.ttf",
    "fonts/JasonWriting2.ttf",
    "fonts/TaipeiSansTCBeta-Bold.ttf",
    "fonts/TaipeiSansTCBeta-Light.ttf",
    "fonts/TaipeiSansTCBeta-Regular.ttf",
    "fonts/aoyagireisyosimo_ttf_2_01.ttf",
    "fonts/KouzanBrushFontSousyo.ttf",
]

bg_generator = BackgroundGenerator(
    "textures",
    zones_range=(10, 30))

combined_datasets = []

font_size = 48
out_size = 128

random_line = RandomLine(font_size, (0, 6), thickness=2, num_samples=500)

post_process = transforms.Compose([
    transforms.ToTensor(),
    random_line,
    RandomDropResolution((20, 45)),
    transforms.Resize((out_size, out_size)),
])

for font_file in font_files:
    filtered_characters = filter_character_list(font_file, characters)
    dirty_datasets = []

    # background + random text color
    dirty_datasets.append(FontDataset(filtered_characters, font_file,
                                      font_size=font_size,
                                      img_size=font_size,
                                      bg_generator=bg_generator,
                                      random_character_color=True,
                                      transform=post_process))

    # random background + pseudo 3D text with random color
    dirty_datasets.append(Pseudo3DFontDataset(filtered_characters, font_file,
                                              font_size=font_size,
                                              img_size=font_size,
                                              bg_generator=bg_generator,
                                              transform=post_process))

    # random background + text with border
    dirty_datasets.append(BorderedFontDataset(filtered_characters, font_file,
                                              font_size=font_size,
                                              img_size=font_size,
                                              bg_generator=bg_generator,
                                              transform=post_process))

    # random background + hollow text
    dirty_datasets.append(HollowFontDataset(filtered_characters, font_file,
                                            font_size=font_size,
                                            img_size=font_size,
                                            bg_generator=bg_generator,
                                            border_size=(1, 2),
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                random_line,
                                                transforms.Resize(
                                                    (out_size, out_size)),
                                            ])))

    # random background + text border + text texture
    dirty_datasets.append(TextureFontDataset(filtered_characters, font_file,
                                             font_size=font_size,
                                             img_size=font_size,
                                             bg_generator=bg_generator,
                                             border_size=(1, 2),
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 random_line,
                                                 transforms.Resize(
                                                     (out_size, out_size)),
                                             ])))

    # random background + double text border + text texture + shadow
    dirty_datasets.append(TextureDBorderFontDataset(filtered_characters, font_file,
                                                    font_size=font_size,
                                                    img_size=font_size,
                                                    bg_generator=bg_generator,
                                                    border_size=(2, 2),
                                                    out_offset=(1, 2),
                                                    shadow_offset_x=(-2, 2),
                                                    shadow_offset_y=(-2, 2),
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        random_line,
                                                        transforms.Resize(
                                                            (out_size, out_size)),
                                                    ])))

    clean_ds = FontDataset(filtered_characters, font_file,
                           font_size=out_size,
                           img_size=out_size,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))

    combined_datasets.append(CombinedDataset(dirty_datasets, clean_ds))


dataset = ConcatDataset(*combined_datasets)
