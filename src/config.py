#上から順に、JOI画像の切り抜き後が入っているもの、コンテストの訓練画像が入っているもの、テスト画像が入っているもの
data_joi_root = '/export/datasets/stakinami/images/joi_fundus_crop/JOI_fundus_crop'
contest_train = '/export/datasets/stakinami/images/fundus_train_cropping/'
contest_test = '/export/datasets/stakinami/images/img_test_crop'

#自前の訓練データ、テストデータのラベルが格納されているファイル
train_info_list = '/export/datasets/stakinami/txt/ageestimate_train.csv'
test_info_list = '/export/datasets/stakinami/txt/ageestimate_test.csv'

#本番のコンテストの訓練データ、テストデータのラベルが格納されているファイル
contest_train_list = '/export/datasets/stakinami/txt/age_all.csv'
contest_test_list = '/export/datasets/stakinami/txt/contest_test.csv'


MODEL_DIR_PATH = './model/'
LOG_DIR_PATH = './log/'
image_size = 224
n_classification = 65