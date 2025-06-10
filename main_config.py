class BaseConfig:
    lr = 0.001
    num_workers = 4
    kd_mode = ['RKD', 'Logits', 'LSKD', 'CTKD', 'DKD', 'CKD']
    save_log = './log'
    indicator_prior = 'OA'
    model_list = [['ResNet', 'ResNet'], ['MobileNet', 'MobileNet'], ['ShuffleNet', 'ShuffleNet'],
                  ['ResNet', 'MobileNet'], ['ResNet', 'ShuffleNet'], ['MobileNet', 'ShuffleNet']]
    train_data_shuffle = True
    test_data_shuffle = False

    dropout = 0.1

    epochs = 200
    batch = 256
    learning_rate = 0.001

    seed = 42
    K_fold = 5

    att_head = 8

    lam_task = 0.3
    lam_kd = 0.4
    lam_ot = 0.3
    ot_weights = 0.01
    att_dim = 512

    cuda_id = "3"
    devices_id = [2, 3]

class ConfigS2SEU(BaseConfig):
    data_name = 'S2S_EU'
    train_teacher_root = '[Your Path]/S2S_EU/MS'
    train_student_root = '[Your Path]/S2S_EU/RGB'
    json_dir = './datasets/s2seu.json'
    dynamic_pair_list = './datasets/Dyn_M_S2S_EU.json'
    log_name = 'log_S2S_EU.txt'

    teacher_list = ['./weights/ResNet_t_S2S_EU.pth',
                    './weights/MobileNet_t_S2S_EU.pth',
                    './weights/ShuffleNet_t_S2S_EU.pth']

    student_list = ['./weights/ResNet_s_S2S_EU.pth',
                    './weights/MobileNet_s_S2S_EU.pth',
                    './weights/ShuffleNet_s_S2S_EU.pth']

    category_list = ['SeaLake', 'River', 'Residential', 'PermanentCrop', 'Pasture', 'Industrial', 'Highway',
                     'HerbaceousVegetation', 'Forest', 'AnnualCrop']
    label_type = "Single"

    train_num_class = 10
    msi_channel = 10
    rgb_channel = 3

    adjust_epoch = [30, 60, 95, 135]

class ConfigS2SCN(BaseConfig):
    data_name = 'S2S_CN'
    train_teacher_root = '[Your Path]/S2S_CN/MS'
    train_student_root = '[Your Path]/S2S_CN/RGB'
    json_dir = './datasets/s2scn.json'
    dynamic_pair_list = './datasets/Dyn_M_S2S_CN.json'
    log_name = 'log_S2S_CN.txt'

    teacher_list = ['./weights/ResNet_t_S2S_CN.pth',
                    './weights/MobileNet_t_S2S_CN.pth',
                    './weights/ShuffleNet_t_S2S_CN.pth']

    student_list = ['./weights/ResNet_s_S2S_CN.pth',
                    './weights/MobileNet_s_S2S_CN.pth',
                    './weights/ShuffleNet_s_S2S_CN.pth']

    category_list = ['beach', 'circularfarmland', 'cloud', 'desert', 'forest',
                     'mountain', 'rectangularfarmland', 'residential', 'river', 'snowberg']
    label_type = "Single"

    train_num_class = 10
    msi_channel = 14
    rgb_channel = 3
    adjust_epoch = [45, 60, 80, 105, 135]

class ConfigM2SGL(BaseConfig):
    data_name = 'M2S_GL'
    train_teacher_root = '[Your Path]/M2S_GL/MS'
    train_student_root = '[Your Path]/M2S_GL/RGB'
    json_dir = './datasets/m2sgl.json'
    dynamic_pair_list = './datasets/Dyn_M_M2S_GL.json'
    log_name = 'log_M2S_GL.txt'

    teacher_list = ['./weights/ResNet_t_M2S_GL.pth',
                    './weights/MobileNet_t_M2S_GL.pth',
                    './weights/ShuffleNet_t_M2S_GL.pth']

    student_list = ['./weights/ResNet_s_M2S_GL.pth',
                    './weights/MobileNet_s_M2S_GL.pth',
                    './weights/ShuffleNet_s_M2S_GL.pth']

    category_list = ['forest', 'agriculture', 'shrub', 'pasture', 'water_body',
                     'sea', 'industry', 'grassland', 'water_course', 'crop',
                     'sport', 'transport', 'beach', 'airport', 'port']

    label_type = "Multi"

    train_num_class = 15
    msi_channel = 10
    rgb_channel = 3

    adjust_epoch = [15, 45, 80, 120, 165]

def run_config(dataset):
    config_map = {
        'S2S_EU': ConfigS2SEU,
        'S2S_CN': ConfigS2SCN,
        'M2S_GL': ConfigM2SGL,
    }

    if dataset not in config_map:
        raise ValueError('Wrong Dataset Type')

    return config_map[dataset]()
