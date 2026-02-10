#Object Detection
#Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import tensorflow as tf
import tensorflow_datasets as tfds
import os

#Visualization Utilities

im_width=75
im_height=75
use_normalized_coordinates=True

def draw_bounding_on_image_array(image,boxes,color=[],thickness=1,display_str_list=()):
    image_pil=PIL.Image.fromarray(image)
    rgbimg=PIL.Image.new('RGBA',image_pil.size)
    rgbimg.paste(image_pil)
    draw_bounding_boxes_on_image(rgbimg,boxes,color,thickness,display_str_list)
    return np.array(rgbimg)
def draw_bounding_boxes_on_image(image,boxes,color=[],thickness=1,display_str_list=()):
    boxes_shape=boxes.get_shape()
    if not boxes_shape:
        return
    if len(boxes_shape)!=2 or boxes_shape[1]!=4:
        raise ValueError('boxes must have shape [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image,boxes[i,1],boxes[i,0],boxes[i,3],boxes[i,2],color[i],thickness,display_str_list[i])
def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color='red',thickness=1,display_str_list=None,use_normalized_coordinates=True):
    draw= PIL.ImageDraw.Draw(image)
    im_width,im_height=image.size
    if use_normalized_coordinates:
        (left,right,top,bottom)=(xmin*im_width,xmax*im_width,
                                 ymin*im_height,ymax*im_height)
    else:
        (left,right,top,bottom)=(xmin,xmax,ymin,ymax)
    draw.line([(left,top),(left,bottom),(right,bottom),(right,top),(left,top)],width=thickness,fill=color)
def datasets_to_numpy_util(training_datasets,validations_datasets,N):
    batch_train_ds=training_datasets.unbatch().batch(N)
    
    if tf.executing_eagerly():
        for validation_digits,(validation_images, validation_labels) in validations_datasets:
            validation_digits=validation_digits.numpy()
            validation_labels=validation_labels.numpy()
            validation_bboxes=validation_bboxes.numpy()
        for training_digits,(training_labels,training_bboxes) in training_datasets:
            training_digits=training_digits.numpy()
            training_labels=training_labels.numpy()
            training_bboxes=training_bboxes.numpy()
    
    validation_labels=np.argmax(validation_labels,axis=1)
    training_labels=np.argmax(training_labels,axis=1)
    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)
MATPLOTLIB_FONT_DIR=os.path.join(os.path.dirname(plt._file_),"mpl-data/fonts/ttf")
def create_digits_from_local_fonts(n):
    font_labels=[]
    img=PIL.Image.new('LA',(75*n,75),color=(0,255))
    font1=PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR,'DejaVuSansMono-Oblique.ttf'),25)
    font2=PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR,'STIXGeneral.ttf'),25)
    d=PIL.ImageDraw.Draw(img)
    for i in range(n):
        font_labels.append(i%10)
        d.text((7+i*75,0 if i<10 else -4), str(i%10),fill=(255,255),font=font1 if i<10 else font2)
    font_digits=np.array(img.getdata(),np.float32)[:,0]/255.0
    font_digits=np.reshape(np.stack(np.split(np.reshape(font_digits,[75,75*n]),n,axis=1),axis=0)[n,75*75])
    return font_digits,font_labels
def display_digits_with_boxes(digits,predictions,labels,pred_bboxes,bboxes,iou,title):
    n=10
    indexes=np.random.choice(len(digits), size=n)
    n_digits=digits[indexes]
    n_predictions=predictions[indexes]
    n_labels=labels[indexes]
    n_iou=[]
    if len(iou)>0:
        n_iou=iou[indexes]
    if len(pred_bboxes)>0:
        n_pred_bboxes=pred_bboxes[indexes]
    if len(bboxes)>0:
        n_bboxes=bboxes[indexes]
    n_digits=n_digits*255.0
    n_digits=n_digits.reshape(n,75,75)
    fig=plt.figure(figsize=(20,4))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    
    for i in range(n):
        ax=fig.add_subplot(1,n,i+1)
        bboxes_to_plot=[]
        if (len(pred_bboxes)>1):
            bboxes_to_plot.append(n_pred_bboxes[i])
            
        if (len(bboxes)>1):
            bboxes_to_plot.append(n_bboxes[i])
        
        img_to_draw= draw_bounding_on_image_array(image=n_digits[i],
                                                  boxes=np.asarray(bboxes_to_plot),
                                                  colour=['red','green'],
                                                  display_str_list=['True','Pred'])
        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])

        if n_predictions[i]!=n_labels[i]:
            ax.xaxis.label.set_color('red')
        plt.imshow(img_to_draw)
        
        if len(iou)>0:
            color='black'
            if n_iou[i][0]<iou_threshold:
                color='red'
            ax.text(0.2,-0.3,'iou: %s'%(n_iou[i][0]),color=color,transform=ax.transAxes,)
def plot_metrics(metric_name,title):
    plt.title(title)
    plt.plot(history.history[metric_name], color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green',label='val_' + metric_name)

#Loading and Preprocessing the Dataset

stratergy=tf.distribute.get_strategy()
stratergy.num_replicas_in_sync
BATCH_SIZE=64*stratergy.num_replicas_in_sync
def read_image_tfds(image,label):
   xmin=tf.random.uniform((),0,48,dtype=tf.int32)
   xmax=tf.random.uniform((),0,48,dtype=tf.int32)
   image=tf.reshape(image(28,28,1))
   image=tf.image.pad_to_bounding_box(image,ymin,ymax,75,75)
   image=tf.cast(image,tf.float32)/255.0
   xmin=tf.cast(xmin,tf.float32)
   ymin=tf.cast(ymin,tf.float32)
   
   xmax=(xmin+28)/75
   ymax=(ymin+4)/75
   xmin=xmin/75
   ymin=ymin/75

   return image, (tf.one_hot(label, 10),[xmin, ymin, xmax, ymax])

def get_training_dataset():
    with stratergy.scope():
        dataset=tfds.load('mnist', split='train', as_supervised=True,try_gcs=True)
        dataset=dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset=dataset.batch(5000,reshuffle_each_iteration=True)
        dataset=dataset.repeat()
        dataset=dataset.batch(BATCH_SIZE,drop_remainder=True)
        dataset=dataset.prefetch(-1)
    return dataset

def get_validation_dataset():
    with stratergy.scope():
        dataset=tfds.load('mnist', split='train', as_supervised=True,try_gcs=True)
        dataset=dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset=dataset.batch(10000,drop_remainder=True)
        dataset=dataset.repeat()
    return dataset
with stratergy.scope():
    training_dataset=get_training_dataset()
    validation_dataset=get_validation_dataset()
(training_digits, training_labels, training_bboxes,
 validation_digits,validation_labels, validation_bboxes) = datasets_to_numpy_util(training_dataset, validation_dataset,0)
display_digits_with_boxes(training_digits, training_labels,
                          training_labels,np.array([]), training_bboxes,
                          np.array([]), 'Training Digits & Lables')
display_digits_with_boxes(validation_digits, validation_labels,
                          validation_labels,np.array([]), validation_bboxes,
                          np.array([]), 'Validation Digits & Lables')

#Define the Network

def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, activation='relu',kernel_size=3,input_shape=(75, 75, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, activation='relu',kernel_size=3)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, activation='relu',kernel_size=3)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    return x
def denser_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x
def classifier(inputs):
    classification_output = tf.keras.layers.Dense(10, activation='softmax', name='classification')(inputs)
    return classification_output
def bounding_box_regression(inputs):
    bounding_box_regression_output = tf.keras.layers.Dense(4, name='bounding_box')(inputs)
    return bounding_box_regression_output
def final_model(inputs):
    feature_cnn=feature_extractor(inputs)
    denser_output=denser_layers(feature_cnn)
    classification_output=classifier(denser_output)
    bounding_box_output=bounding_box_regression(denser_output)
    model=tf.keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])
    return model
def define_and_compile_model(inputs):
    model = final_model(inputs)
    model.compile(optimizer='adam', loss={'classification': 'categorical_crossentropy', 
                                          'bounding_box': 'mse'},
                  metrics={'classification':'accuracy','bounding_box':'mse'})
    return model
with stratergy.scope():
    inputs = tf.keras.Input(shape=(75, 75, 1))
    model = define_and_compile_model(inputs)
model.summary()

#Train and Validatte the Model

EPOCHS = 10
steps_per_epoch = 60000 // BATCH_SIZE

history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset, validation_steps=1, epochs=EPOCHS)

loss, classifier_loss,bounding_box_loss,classifier_acc,bounding_box_mse = model.evaluate(validation_dataset, steps=1)
print("\n--------------------------------\n")
print('Validation Accuracy:', classifier_acc)
print("\n------------------\n")

plot_metrics("bounding_box_mse","Bounding Box MSE")
plot_metrics("classifier_accuracy","Classifier Accuracy")
plot_metrics("classifier_loss","Classifier Loss")

def intersection_over_union(pred_box,true_box):
    xmin_pred,ymin_pred,xmax_pred,ymax_pred=np.split(pred_box,4,axis=-1)
    xmin_true,ymin_true,xmax_true,ymax_true=np.split(true_box,4,axis=-1)
 
    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap),0) * np.maximum((ymax_overlap - ymin_overlap),0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    iou=(overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    
    return iou
prediction = model.predict(validation_dataset, batch_size=64)
predicted_labels = np.argmax(prediction[0], axis=-1)
prediction_bboxes = prediction[1]
iou = intersection_over_union(prediction_bboxes, validation_bboxes)
iou_threshold = 0.6
display_digits_with_boxes(validation_digits,predicted_labels,validation_labels,
                          prediction_bboxes,iou,"True and Pred value")
