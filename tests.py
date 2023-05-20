from pathlib import Path

def test_CNN():
    import numpy as np
    from keras.models import load_model
    from DataGenerator import DataGenerator

    path = Path('Built_dataset')
    cnn_name = 'B_ISIC2017'
    # Predict on test dataset
    path_dataset_test_1 = path / 'data1' / 'test'
    path_dataset_test_2 = path / 'data2' / 'test'
    test_gen = DataGenerator(path_dataset_test_1, path_dataset_test_2)

    # Load CNN if already trained
    cnn_model = load_model(f'models\\{cnn_name}.h5')
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

    path = Path('Built_dataset')
    cnn_name = "B_ISIC2017"

    cnn_model = CNN("default")
    cnn_model.set_epochs(100)
    cnn_model.display(name = cnn_name, graph = True)

    path_dataset_1 = path / 'data1' / 'train'
    path_dataset_2 = path / 'data2' / 'train'

    train_gen = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='train')
    validation = DataGenerator(path_dataset_1, path_dataset_2, seed=1, validation_split=0.2, subset='validation')

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
    from Dataset import Dataset

    original_dataset_path = Path("D:") / 'THESE' / 'CODE' / 'Doctorat' / 'ISIC2017' / 'train'
    # dataset = Dataset("D:\\THESE\\CODE\\Doctorat\\ISIC2017\\train")
    dataset = Dataset(original_dataset_path)
    dataset.process()

def compute_performances():
    from performance import compute_performance
    from skimage import io
    import os 
    import csv

    ground_truths_path = Path('gts')
    results_path = Path('res')

    file_headers = ['img', 'sensitivity', 'Specificity', 'Accuracy', 'jaccard_index', 'dice_coef']
    results_per_model = [dir for dir in os.listdir(results_path) if (results_path/dir).is_dir()]

    for model_results in results_per_model:
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
            # print(f"{res} -- {gt}\n  {accuracy}\n\n")
        
        output_path = results_path / model_results / 'performance'
        output_path.mkdir(exist_ok=True)
        with open(output_path /"results.csv", "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = file_headers)
            writer.writeheader()
            writer.writerows(results_dict)

# create_dataset()
train_CNN()
# test_CNN()
# compute_performances()