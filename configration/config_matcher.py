import torch

class BaseConfig:
    debug = True
    batch = 64
    num_workers = 0
    lr = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_embedding = 512
    text_embedding = 512

    pretrained = False  # for both MS encoder and RGB encoder
    trainable = False  # for both MS encoder and RGB encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both MS and RGB encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

    save_log = './log'
    log_name = 'S2S_EU_CLIP.txt'
    seed = 42

    train_data_shuffle = True
    test_data_shuffle = False

    cuda_id = "3"

    folds = 5
    num_workers = 4

    rgb_channel = 3

    # save_model_path = './weights/best.pt'

class ConfigS2SEU(BaseConfig):
    data_name = 'S2S_EU'
    train_teacher_root = '[Your Path]/S2S_EU/MS'
    unpaired_root = '[Your Path]/S2S_EU/RGB'
    train_num_class = 10
    msi_channel = 10
    category_list = ['SeaLake', 'River', 'Residential', 'PermanentCrop', 'Pasture', 'Industrial', 'Highway',
                     'HerbaceousVegetation', 'Forest', 'AnnualCrop']

    save_model_path = './weights/best_s2s_eu.pt'
    save_json_path = './datasets/s2seu.json'

class ConfigS2SCN(BaseConfig):
    data_name = 'S2S_CN'
    train_teacher_root = '[Your Path]/S2S_CN/MS'
    unpaired_root = '[Your Path]/S2S_CN/RGB'
    train_num_class = 10
    msi_channel = 14
    category_list = ['beach', 'circularfarmland', 'cloud', 'desert', 'forest', 'mountain',
                     'rectangularfarmland', 'residential', 'river', 'snowberg']
    save_model_path = './weights/best_s2s_cn.pt'
    save_json_path = './datasets/s2scn.json'

class ConfigM2SGL(BaseConfig):
    data_name = 'M2S_GL'
    train_teacher_root = '[Your Path]/M2S_GL/MS'
    unpaired_root = '[Your Path]/M2S_GL/RGB'
    train_num_class = 15
    msi_channel = 10
    category_list = ['forest', 'agriculture', 'shrub', 'pasture', 'water_body',
                     'sea', 'industry', 'grassland', 'water_course', 'crop',
                     'sport', 'transport', 'beach', 'airport', 'port']

    save_model_path = './weights/best_m2s_gl.pt'
    save_json_path = './datasets/m2sgl.json'

def run_config(dataset):
    config_map = {
        'S2S_EU': ConfigS2SEU,
        'S2S_CN': ConfigS2SCN,
        'M2S_GL': ConfigM2SGL,
    }

    if dataset not in config_map:
        raise ValueError('Wrong Dataset Type')

    return config_map[dataset]()
