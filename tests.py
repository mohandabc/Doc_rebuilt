from pathlib import Path

def test_CNN():
    import numpy as np
    from keras.models import load_model
    from DataGenerator import DataGenerator

    path = Path('Built_dataset')
    cnn_name = 'RGB_model_epoch_12_val_loss_0.26'
    # Predict on test dataset
    path_dataset_test_1 = path / 'data1' / 'test'
    path_dataset_test_2 = path / 'data2' / 'test'
    test_gen = DataGenerator(path_dataset_test_1, path_dataset_test_2)

    # Load CNN if already trained
    cnn_model = load_model(f'models\\models_to_test\\{cnn_name}.h5')
    res = cnn_model.predict(test_gen)


    # Display results
    rounded = np.argmax(res, axis=-1)
    real = []
    for i in test_gen:  
        real.extend(i[1].numpy())

    s=0
    for i in range(len(real)):
        if rounded[i] == real[i]:
            s+=1
        # print(rounded[i], "=", real[i])
    print(f"result {s} / {len(rounded)} = {s/len(rounded)}")


def train_CNN():
    from CNN import CNN
    from DataGenerator import DataGenerator
    import matplotlib.pyplot as plt
    import pickle

    path = Path('HAM10000')
    cnn_name = "B_HAM10000"

    cnn_model = CNN(cnn_name)
    cnn_model.set_epochs(10)
    cnn_model.set_batch_size(1024)
    cnn_model.display(name = cnn_name, graph = True)

    path_dataset_1 = path / 'data1' / 'train'
    path_dataset_2 = path / 'data2' / 'train'

    train_gen = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='train')
    # train_gen.set_data_type('RGB')
    validation = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='validation')
    # validation.set_data_type('RGB')

    history = cnn_model.train(train_gen, validation)
    with open('./models/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    cnn_model.save(cnn_name)


    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def create_dataset():
    """Our CNN works on windows of the image (parts) and since we want them to be reusable
    we will generate them once and save them as a dataset. So we will not work with original 
    dataset but with the generated on
    """
    from Dataset import Dataset

    # original_dataset_path = Path("D:") / 'THESE' / 'CODE' / 'Doctorat' / 'ISIC2017' / 'train'
    original_dataset_path = Path("D:") / 'THESE' / 'CODE' / 'Doctorat' / 'all_datasets' / 'HAM10000'
    dataset = Dataset(original_dataset_path, output_folder="HAM10000", images_folder="HAM10000_images", masks_folder="HAM10000_segmentations_lesion")
    dataset.process()

def compute_performances(res='res', gts='gts'):
    from performance import compute_performance
    from skimage import io
    import os 
    import csv

    ground_truths_path = Path(gts)
    results_path = Path(res)

    file_headers = ['img', 'sensitivity', 'Specificity', 'Accuracy', 'jaccard_index', 'dice_coef']
    results_per_model = [dir for dir in os.listdir(results_path) if (results_path/dir).is_dir()]

    for model_results in results_per_model:
        _sum = {'sensitivity' : 0, 'Specificity' : 0, 'Accuracy' : 0, 'jaccard_index' : 0, 'dice_coef' : 0}
        _max = {'sensitivity' : 0, 'Specificity' : 0, 'Accuracy' : 0, 'jaccard_index' : 0, 'dice_coef' : 0}
        _min = {'sensitivity' : 1, 'Specificity' : 1, 'Accuracy' : 1, 'jaccard_index' : 1, 'dice_coef' : 1}
        print(f"------------------{model_results}-------------------\n")
        results_dict = []
        results = [img for img in os.listdir(results_path/model_results) if not (results_path/model_results/img).is_dir()]
        # print(results)
        gts = [img for img in os.listdir(ground_truths_path)]
        i = 1
        for res, gt in zip(results, gts):
            print(f'{i}-   {res} - {gt}\n')
            i+=1
            res_path = results_path / model_results / res
            gt_path = ground_truths_path / gt

            res_img = io.imread(res_path)
            gt_img = io.imread(gt_path)

            accuracy = compute_performance(res_img, gt_img)
            accuracy['img'] = f"{res} -- {gt}"
            results_dict.append(accuracy)

            _sum['sensitivity'] += results_dict[-1]['sensitivity']
            _sum['Specificity'] += results_dict[-1]['Specificity']
            _sum['Accuracy'] += results_dict[-1]['Accuracy']
            _sum['jaccard_index'] += results_dict[-1]['jaccard_index']
            _sum['dice_coef'] += results_dict[-1]['dice_coef']

            _max['sensitivity'] = max(_max['sensitivity'], results_dict[-1]['sensitivity'])
            _max['Specificity'] = max(_max['Specificity'], results_dict[-1]['Specificity'])
            _max['Accuracy'] = max(_max['Accuracy'], results_dict[-1]['Accuracy'])
            _max['jaccard_index'] = max(_max['jaccard_index'], results_dict[-1]['jaccard_index'])
            _max['dice_coef'] = max(_max['dice_coef'], results_dict[-1]['dice_coef'])

            _min['sensitivity'] = min(_min['sensitivity'], results_dict[-1]['sensitivity'])
            _min['Specificity'] = min(_min['Specificity'], results_dict[-1]['Specificity'])
            _min['Accuracy'] = min(_min['Accuracy'], results_dict[-1]['Accuracy'])
            _min['jaccard_index'] = min(_min['jaccard_index'], results_dict[-1]['jaccard_index'])
            _min['dice_coef'] = min(_min['dice_coef'], results_dict[-1]['dice_coef'])
            # print(f"{res} -- {gt}\n  {accuracy}\n\n")
        
        _mean = {}
        _mean['sensitivity'] = _sum['sensitivity'] / len(results) 
        _mean['Specificity'] = _sum['Specificity'] / len(results) 
        _mean['Accuracy'] = _sum['Accuracy'] / len(results) 
        _mean['jaccard_index'] = _sum['jaccard_index'] / len(results) 
        _mean['dice_coef'] = _sum['dice_coef'] / len(results) 
        
        _max['img'] = "Max"
        _min['img'] = "Min"
        _mean['img'] = "Mean"

        results_dict.insert(0, _min)
        results_dict.insert(0, _max)
        results_dict.insert(0, _mean)
    
        output_path = results_path / model_results / 'performance'
        output_path.mkdir(exist_ok=True)
        with open(output_path /"results.csv", "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = file_headers)
            writer.writeheader()
            writer.writerows(results_dict)

def segmentation():
    # In this file we will use the trained model to segment, then perform a diagnosis
    from skimage import io, img_as_ubyte
    from Segmentation import Segmentation
    import os
    from pathlib import Path

    models_path = Path("models") /"models_to_test"
    models_to_test = [model for model in os.listdir(models_path) if not (models_path/model).is_dir()]
    # imgs_path = Path("D:/THESE/CODE/Doctorat/all_datasets/HAM10000/HAM10000_images")
    imgs_path = Path("img_to_test")

    results_path = Path("res")
    if not results_path.exists():
        os.mkdir(Path('res'))

    for model in models_to_test:
        model_name = model.split('.')[0]
        data_type = model_name.split('_')[0]
        model_path = models_path / model

        print(f"---------------- USING MODEL {model}-------------------\n")
        model_result_path = results_path / model_name
        if not model_result_path.exists():
            os.mkdir(model_result_path)
        if not (model_result_path / 'EXTRA').exists():
            os.mkdir(model_result_path / 'EXTRA')

        s = Segmentation(imgs_path, model_path, data_type)
        seg_results = s.segmentation()

        print('Start Segmentation...........\n')
        for seg, name, sp1, sp2 in seg_results:
            print('Segmentation over ...........\n')
            
            res_path = model_result_path / name
            io.imsave(res_path, img_as_ubyte(seg))
            io.imsave(model_result_path / 'EXTRA' / f'slic_{name}', img_as_ubyte(sp1))
            io.imsave(model_result_path / 'EXTRA' / f'wat_{name}', img_as_ubyte(sp2))
            print('Start Segmentation...........\n')

def fill_holes():
    import os 
    from skimage import io
    from utils import fill_holes

    mask_path = Path('res')  / 'B_model_epoch_24_val_loss_0'
    output_path = Path('filled') / '3000'
    output_path.mkdir(exist_ok=True)

    masks = [img for img in os.listdir(mask_path) if not (mask_path/img).is_dir()]

    i = 1
    for mask in masks:
        print(f'{i}------{mask}------')
        i+=1
        img = io.imread(mask_path / mask)
        filled_mask=  fill_holes(img)
        io.imsave(output_path / mask, filled_mask)


"""
Comment or uncomment one of these function calls to execute a code
"""

# create_dataset()
# train_CNN()
# test_CNN()
segmentation()
compute_performances()
fill_holes()
compute_performances('filled/3000', 'gts')