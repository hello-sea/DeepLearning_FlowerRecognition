
import numpy as np
import tensorflow as tf
import pickle as pickle # python pkl 文件读写

from CNNs_model import cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):


    predict_data = np.array(pickle.load(open('cache/test_temporary_data.plk', 'rb')) )
    predict_labels = np.array(pickle.load(open('cache/test_temporary_labels.plk', 'rb')) )

    # predict_data = np.array(pickle.load(open('cache/test_data.plk', 'rb')) )
    # predict_labels = np.array(pickle.load(open('cache/test_labels.plk', 'rb')) )

    # with tf.Session() as sess:
    #     train_data = tf.convert_to_tensor(train_data_np)
    #     eval_data = tf.convert_to_tensor(eval_data_np)

    # Create the Estimator
    cnn_classifier = tf.estimator.Estimator(
        # model_fn=cnn_model_fn, model_dir="cnn_convnet_model")
        model_fn=cnn_model_fn, model_dir="Model/cnn")


    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        y=predict_labels,
        num_epochs=1,
        shuffle=False) 
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        num_epochs=1,
        shuffle=False) 
    predict_results = cnn_classifier.predict(input_fn=predict_input_fn)


    for e in  predict_results:
        print(e['classes'])
        # print(e['probabilities'])
    print("done!")

    for i in predict_labels:
        print(i)




if __name__ == "__main__":
    
    tf.app.run()


