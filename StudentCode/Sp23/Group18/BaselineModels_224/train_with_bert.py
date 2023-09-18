
#importing packages
import warnings
warnings.filterwarnings("ignore")

import pickle
import os
import pandas as pd
import random as rn
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
import datetime

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.regularizers import l1,l2
from tqdm import tqdm
import heapq
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop

# In[3]:


import bert

model_name = "uncased_L-12_H-768_A-12"
model_dir = bert.fetch_google_bert_model(model_name, ".models")
model_ckpt = os.path.join(model_dir, "bert_model.ckpt")

bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")


# In[5]:

currentDirectory = "./"
dataDirectory = "../"
imageDirectory = dataDirectory + "image_data/train2014/"
val_imageDirectory = dataDirectory + "image_data/val2014/"
modelsDirectory = currentDirectory + "bert_models/"
img_width = 224
img_height = 224

BATCH_SIZE = 64
BUFFER_SIZE = 300


# In[6]:

data = pd.read_csv(dataDirectory + 'image_text_data.csv')
val_data = pd.read_csv(dataDirectory + 'val_image_text_data.csv')

data.dropna(inplace=True)
val_data.dropna(inplace=True)


# In[7]:


contractions = { 
"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not",
"couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he's": "he is","how'd": "how did",
"how'll": "how will","how's": "how is","i'd": "i would","i'll": "i will","i'm": "i am","i've": "i have","isn't": "is not","it'd": "it would",
"it'll": "it will","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
"must've": "must have","mustn't": "must not","needn't": "need not","oughtn't": "ought not","shan't": "shall not","sha'n't": "shall not","she'd": "she would",
"she'll": "she will","she's": "she is","should've": "should have","shouldn't": "should not","that'd": "that would","that's": "that is","there'd": "there had",
"there's": "there is","they'd": "they would","they'll": "they will","they're": "they are","they've": "they have","wasn't": "was not","we'd": "we would",
"we'll": "we will","we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what're": "what are","what's": "what is",
"what've": "what have","where'd": "where did","where's": "where is","who'll": "who will","who's": "who is","won't": "will not","wouldn't": "would not",
"you'd": "you would","you'll": "you will","you're": "you are"
}

def preprocess_text(text):
    '''Given a text this function removes the punctuations and returns the remaining text string'''
    new_text = ""
    text = text.lower()
    i = 0
    for word in text.split():
        if i==0:
            new_text = contractions.get(word,word)
        else:
            new_text = new_text + " " + contractions.get(word,word)
        i += 1
    return new_text.replace("'s", '')


# In[8]:


data['multiple_choice_answer'] = data['multiple_choice_answer'].apply(lambda x: preprocess_text(x))
val_data['multiple_choice_answer'] = val_data['multiple_choice_answer'].apply(lambda x: preprocess_text(x))


# In[9]:

# X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)
# len(X_train), len(X_val)
X_train = data
X_val = val_data
del data, val_data
len(X_train), len(X_val)


# In[10]:


all_classes = X_train['multiple_choice_answer'].values
class_frequency = {}

for _cls in all_classes:
    if(class_frequency.get(_cls, -1) > 0):
        class_frequency[_cls] += 1
    else:
        class_frequency[_cls] = 1


# In[11]:


sort_class_frequency = sorted(list(class_frequency.items()), key = lambda x: x[1],reverse=True)   

plt.plot([x[1] for x in sort_class_frequency[:30]])
i=np.arange(30)
plt.title("Frequency of top 30 Classes")
plt.xlabel("Tags")
plt.ylabel("Counts")
plt.xticks(i,[x[0] for x in sort_class_frequency[:30]])
plt.xticks(rotation=90)
plt.show()


# In[12]:


common_tags = heapq.nlargest(1000, class_frequency, key = class_frequency.get)


# In[13]:


# take the top 1000 classes
X_train['multiple_choice_answer'] =  X_train['multiple_choice_answer'].apply(lambda x: x if x in common_tags else '')

# removing question which has empty tags
X_train = X_train[X_train['multiple_choice_answer'].apply(lambda x: len(x)>0)]


# In[14]:


label_encoder = LabelBinarizer()
answer_vector_train = label_encoder.fit_transform(X_train['multiple_choice_answer'].apply(lambda x: x).values)
answer_vector_val = label_encoder.transform(X_val['multiple_choice_answer'].apply(lambda x: x).values)

ans_vocab = {l: i for i, l in enumerate(label_encoder.classes_)}

print("Number of clasess: ", len(ans_vocab))
print("Shape of Answer Vectors in Train Data: ", answer_vector_train.shape)
print("Shape of Answer Vectors in Validation Data: ", answer_vector_val.shape)


# ### Question vectors

# In[15]:

def preprocess_question(text):
    '''Given a text this function removes the punctuations and returns the remaining text string'''
    new_text = "<start>"
    text = text.lower()
    for word in text.split():
        new_text = new_text + " " + contractions.get(word, word)
    new_text = new_text + " <end>"
    return new_text.replace("'s", '')


# In[16]:


X_train['question'] = X_train['question'].apply(lambda x: preprocess_question(x))
X_val['question'] = X_val['question'].apply(lambda x: preprocess_question(x))


# In[17]:


# !pip install pytorch_pretrained_bert pytorch-nlp


# In[18]:


from pytorch_pretrained_bert import BertTokenizer

# BertTokenizer = bert.bert_tokenization.FullTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_question_seqs = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:24] + ['[SEP]'], X_train['question'].tolist()))
val_question_seqs = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:24] + ['[SEP]'], X_val['question'].tolist()))

question_vector_train = tf.keras.preprocessing.sequence.pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_question_seqs)), maxlen=24, truncating="post", padding="pre", dtype="int")
question_vector_val = tf.keras.preprocessing.sequence.pad_sequences(list(map(tokenizer.convert_tokens_to_ids, val_question_seqs)), maxlen=24, truncating="post", padding="pre", dtype="int")


print("Shape of Question Vectors in Train Data: ", question_vector_train.shape)
print("Shape of Question Vectors in Validation Data: ", question_vector_val.shape)


# ### Image features

# In[20]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_width, img_height))
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img = img * (1./255)
    img = tf.expand_dims(img, axis=0)
    return img, image_path

def VGG19_Top():
    model = tf.keras.applications.VGG19(include_top=False,weights='imagenet',input_shape = (img_width, img_height,3))
    input_layer = model.input
    hidden_layer = model.layers[-1].output 
    model = tf.keras.Model(input_layer, hidden_layer)
    return model

def generateImageFeatures(images):
    model = VGG19_Top()
    all_image_dict = {}
    img_ds = tf.data.Dataset.from_tensor_slices(images)
    img_ds = img_ds.map(load_image)

#     for img, path in img_ds:
#         img_features = model(img)
#         image_path = path.numpy().decode('utf-8')
#         all_image_dict[image_path] = img_features.numpy()

    for batch_img, batch_path in img_ds:
        batch_img_features = model(batch_img)

        for img_features, path in zip(batch_img_features, batch_path):
            image_path = path.numpy().decode("utf-8")
            #image_path = image_path.replace(imageDirectory,imageNumpyDirectory).replace('.jpg',"")
            #np.save(image_path, img_features.numpy())
            all_image_dict[image_path] = img_features.numpy()
            
    with open(dataDirectory + 'all_image_dict.pickle', 'wb') as handle:
        pickle.dump(all_image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


# In[21]:


# all_image_path = data['image_id'].apply(lambda x:  imageDirectory + 'COCO_train2014_' + '%012d.jpg' % (x)).unique()
# generateImageFeatures(all_image_path)


# In[22]:


image_paths_train = X_train['image_id'].apply(lambda x:  imageDirectory + 'COCO_train2014_' + '%012d.jpg' % (x)).values
image_paths_val = X_val['image_id'].apply(lambda x:  val_imageDirectory + 'COCO_val2014_' + '%012d.jpg' % (x)).values

with open(dataDirectory + 'all_image_dict.pickle', 'rb') as handle:
    all_image_dict = pickle.load(handle)
    
with open(dataDirectory + 'val_image_dict.pickle', 'rb') as handle:
    val_image_dict = pickle.load(handle)
    
all_image_dict.update(val_image_dict)
del val_image_dict


kv = list(all_image_dict.items())
all_image_dict.clear()

for k, v in kv:
    new_key = "." + k
    all_image_dict[new_key] = v

# ### Dataset

# In[23]:


def get_imageTensor(img, ques):
    img_tensor = all_image_dict[img.decode('utf-8')]
    img_tensor = tf.squeeze(img_tensor)
    return img_tensor, ques

def createDataset(image_paths, question_vector, answer_vector):
    dataset_input = tf.data.Dataset.from_tensor_slices((image_paths, question_vector.astype(np.float32)))
    print('ques : ' + str(question_vector.shape))
    dataset_output = tf.data.Dataset.from_tensor_slices((answer_vector.astype(np.float32)))
    # using map to load the numpy files in parallel
    dataset_input = dataset_input.map(lambda img, ques : tf.numpy_function(get_imageTensor, [img, ques], [tf.float32, tf.float32]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffling and batching
    #dataset_input = dataset_input.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_input = dataset_input.batch(BATCH_SIZE)
    dataset_output = dataset_output.batch(BATCH_SIZE)#.repeat()
    
    dataset = tf.data.Dataset.zip((dataset_input, dataset_output))
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return dataset


# In[24]:


dataset_train = createDataset(image_paths_train, question_vector_train, answer_vector_train)
dataset_val = createDataset(image_paths_val, question_vector_val, answer_vector_val)


# ### Baseline model

# In[25]:


##fixing numpy RS
np.random.seed(42)

##fixing tensorflow RS
tf.random.set_seed(32)

##python RS
rn.seed(12)


# In[28]:


def callBacksList():
    """
    returns list of callback's
    """
    filepath = modelsDirectory + ModelName + "/final_best_50ep.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 3)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 5, verbose = 1)

    #directory for tensorboard to save evnts
    # log_dir= modelsDirectory + "logs/fit/" + ModelName + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # os.makedirs(log_dir)

    # print("TensorBoard Folder for this Execution",log_dir)#creating TensorBoard call back,this will write all events to given folder
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

    history = tf.keras.callbacks.History()
    # callbacks_list = [history, tensorboard_callback, checkpoint]
    callbacks_list = [reduce_lr, history, checkpoint]
    
    return callbacks_list

def Build_BaseModel():
    l_bert.trainable = False
    
    image_input = tf.keras.layers.Input(shape=(7, 7, 512))
    question_input = tf.keras.layers.Input(shape=(question_vector_train.shape[1],))

    image_flatten = tf.keras.layers.Flatten()(question_input)

    # Input 2 Pathway
    question_emb = l_bert(question_input)

    question_flatten = tf.keras.layers.Flatten(name="Flatten_lstm")(question_emb)

    image_question = tf.keras.layers.concatenate([image_flatten, question_flatten])

    bn = tf.keras.layers.BatchNormalization()(image_question)
    dropout = tf.keras.layers.Dropout(0.5)(bn)
    
    output = tf.keras.layers.Dense(len(ans_vocab), activation='softmax', 
                                   kernel_initializer = tf.keras.initializers.he_normal(seed=42))(dropout)

    # Create Model
    model = tf.keras.models.Model(inputs = [image_input, question_input], outputs = output)
    
    return model


# In[29]:


l2_alpha = 0.001
ModelName = "Final_bertModel"
model = Build_BaseModel()


# In[30]:


bert.load_bert_weights(l_bert, model_ckpt)


# In[31]:


model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# In[32]:


# tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


history = model.fit(dataset_train, 
                    epochs = 50, 
                    validation_data = dataset_val, 
                    callbacks = callBacksList())

model.save('./final_models/final_bert_model_50epochs.h5')

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'final_history/final_epoch_bert_history_50ep.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

