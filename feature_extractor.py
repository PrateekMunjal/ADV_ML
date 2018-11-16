import numpy as np
import tensorflow as tf
from scipy.misc import imsave
import sys
import matplotlib.pyplot as plt
import random

tf.set_random_seed(0);
random.seed(0);

inpt = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name="input")
domain = tf.placeholder(tf.float32, shape=[None, 1], name="domain")
src_labels = tf.placeholder(tf.float32,shape=[None,10]);
target_labels = tf.placeholder(tf.float32,shape=[None,10]);

#DCGAN 
initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02);#tf.contrib.layers.xavier_initializer()

#tuning knobs
z_dim = 128;
enc1_z = enc2_z = int(z_dim/2.0);

def leaky_relu(x):
    return tf.nn.leaky_relu(x)

def encoder1(x,isTrainable=True,reuse=False):
    with tf.variable_scope("encoder1") as scope:
        if reuse:
            scope.reuse_variables();

        #32x32x1
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #16x16x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        #8x8x128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #4x4x256
        
        fc = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc, units = enc1_z, 
            kernel_initializer= initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

def encoder2(x,isTrainable=True,reuse=False):
    with tf.variable_scope("encoder2") as scope:
        if reuse:
            scope.reuse_variables();

        #32x32x1
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #16x16x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        #8x8x128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=leaky_relu,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #4x4x256
        
        fc = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc, units = enc1_z, 
            kernel_initializer= initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1

def decoder(z,isTrainable=True,reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables();

        fc = tf.layers.dense(z, units = 4*4*512, 
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse)
        fc = tf.layers.batch_normalization(fc,name='fc_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        fl = tf.reshape(fc, [-1, 4, 4, 512]);

        #4x4x512
        deconv1 = tf.layers.conv2d_transpose(fl, 256, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv1_layer')
        deconv1 = tf.layers.batch_normalization(deconv1,name='deconv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        #print("dec_deconv1.shape : ",deconv1.shape);

        #8x8x256
        deconv2 = tf.layers.conv2d_transpose(deconv1, 128, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv2_layer')
        deconv2 = tf.layers.batch_normalization(deconv2,name='deconv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);


        #16x16x128
        deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv3_layer')
        deconv3 = tf.layers.batch_normalization(deconv3,name='deconv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);


        #32x32x64
        deconv4 = tf.layers.conv2d_transpose(deconv3, 1, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv4_layer')
        ##Now the o/p is 32x32x1

        #deconv4 = tf.layers.batch_normalization(deconv4,name='deconv4_layer_batchnorm',
            #trainable=isTrainable,reuse=reuse);

        #32x32x1
        #deconv5 = tf.layers.conv2d_transpose(deconv4, 5, 1, strides=1, padding="SAME", 
         #   kernel_initializer=initializer, activation = tf.nn.tanh,trainable=isTrainable,reuse=reuse,name='')

        return deconv4

def classifier(enc_ft,isTrainable=True,reuse=False):
    with tf.variable_scope("classifier") as scope:
        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(enc_ft, units = 64,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1_layer')
        fc1 = tf.layers.batch_normalization(fc1,name='classifier_fc1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc2 = tf.layers.dense(fc1, units = 32,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2_layer')
        fc2 = tf.layers.batch_normalization(fc2,name='classifier_fc2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc3 = tf.layers.dense(fc2, units = 16,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3_layer')
        fc3 = tf.layers.batch_normalization(fc3,name='classifier_fc3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
            
        fc4 = tf.layers.dense(fc2, units = 10,
            kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse,name='classifier_fc3_layer') 

        return fc4;

def discriminator(x,isTrainable=True,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables();

        # disc1 = tf.layers.conv2d(x, 256, 4, strides=2, padding="SAME", 
        #   kernel_initializer=initializer, activation = leaky_relu)
        # disc2 = tf.layers.conv2d(disc1, 256, 4, strides=2, padding="SAME", 
        #   kernel_initializer=initializer, activation = leaky_relu)
        # disc3 = tf.layers.conv2d(disc2, 256, 4, strides=2, padding="SAME", 
        #   kernel_initializer=initializer, activation = leaky_relu)
        # fc_disc = tf.contrib.layers.flatten(disc3)
        fc1 = tf.layers.dense(x, units = 64,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc1_layer')
        fc1 = tf.layers.batch_normalization(fc1,name='fc1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc2 = tf.layers.dense(fc1, units = 32,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc2_layer')
        fc2 = tf.layers.batch_normalization(fc2,name='fc2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc3 = tf.layers.dense(fc2, units = 16,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc3_layer')
        fc3 = tf.layers.batch_normalization(fc3,name='fc3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        fc4 = tf.layers.dense(fc3, units = 4,
            kernel_initializer = initializer, activation = leaky_relu,trainable=isTrainable,reuse=reuse,name='fc4_layer')
        fc4 = tf.layers.batch_normalization(fc4,name='fc4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        logits = tf.layers.dense(fc4, units = 1,
            kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse,name='fc_logits_layer')

        return logits



enc1 = encoder1(inpt)
enc2 = encoder2(inpt)

disc1 = discriminator(enc1)
disc2 = discriminator(enc2,reuse=True)

fol = tf.nn.sigmoid(disc1)
n_fol = tf.nn.sigmoid(disc2)



z_concat = tf.concat([enc1,enc2],axis=1)
dec = decoder(z_concat)

print('z_concat.shape : ',z_concat.shape);

######## OPS FOR INFERENCING
enc2_test = encoder2(inpt,isTrainable=False,reuse=True);
enc1_test = encoder1(inpt,isTrainable=False,reuse=True);

z_concat_test = tf.concat([enc1_test,enc2_test],axis=1);

print('z_concat_test.shape : ',z_concat_test.shape);

dec_test = decoder(z_concat_test,isTrainable=False,reuse=True);

logits_clf = classifier(enc1_test);

test_logits_clf = classifier(enc1_test,isTrainable=False,reuse=True);

clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_clf,labels=src_labels));

#test_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_logits_clf,labels=target_labels));

target_class = tf.argmax(target_labels,1);
predicted_class = tf.argmax(tf.nn.sigmoid(test_logits_clf),1);
train_class = tf.argmax(src_labels,1);


train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(predicted_class,train_class))) 
test_accuracy = tf.reduce_mean(tf.to_float(tf.equal(predicted_class,target_class)))


disc_invariant_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc1, labels=domain))
disc_variant_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc2, labels=domain))


enc1_invariant_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc1, labels=0.5*tf.ones_like(domain)))

recon_loss = tf.reduce_mean(tf.square(inpt - dec))

#combining the disc for both var and invae enc via beta
beta = 0.1;
var_encoder_loss = beta*disc_variant_loss + (1-beta)*recon_loss;



discriminator_loss = disc_invariant_loss + disc_variant_loss;

#combined fwdkl nd revkl via alpha
alpha = 0.1;
inv_encoder_loss = alpha*enc1_invariant_loss + (1-alpha)*recon_loss;

variables = tf.trainable_variables()

decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder');
invariant_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder1');
variant_enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder2');
discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator');
classifier_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='classifier');

#recon_vars = [var for var in variables if ("discriminator" not in var.name)]

#disc_vars = [var for var in variables if ("discriminator" in var.name)]

#invariant_enc_vars = [var for var in variables if ("encoder1" in var.name)]

#variant_vars = [var for var in variables if ("encoder2" in var.name or "discriminator" in var.name)]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):

    recon_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(recon_loss, var_list = decoder_params);
    #inv_disc : for inv encoder the discriminator
    disc_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(discriminator_loss, var_list = discriminator_params)
    enc_inv_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(inv_encoder_loss, var_list = invariant_enc_params)
    enc_var_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(var_encoder_loss, var_list = variant_enc_params)
    clf_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(clf_loss, var_list = classifier_params);


tf.summary.scalar("recon_loss", recon_loss)
tf.summary.scalar("discriminator_loss_for_variant_enc", disc_variant_loss)
tf.summary.scalar("discriminator_loss_for_invariant_enc", disc_invariant_loss)
tf.summary.scalar("discriminator loss",discriminator_loss);
#tf.summary.scalar("variant_loss", disc_variant_loss)
tf.summary.scalar("enc_inv_loss -- wants to confuse discriminator", inv_encoder_loss)
tf.summary.scalar("enc_var_loss -- wants to help discriminator", var_encoder_loss)


merged_all = tf.summary.merge_all()

log_dir = "feature_extractr_3"




init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



clf_train_epochs = 50; 
epochs = 100;
batch_size = 128

checkpoint_path = "./feature_extractr_3"

mode = 'train';

#mode = 'test';

if mode == 'train':

    with tf.Session() as sess:
        sess.run(init)
        
        source = np.load("dataset/mnist.npy")
        target = np.load("dataset/svhn.npy");
        # #target = np.load("usps.npy")

        print('source.shape : ',source.shape);
        print('target.shape : ',target.shape);

        # print('np.amin(source) : ',np.amin(source));
        # print('np.amax(source) : ',np.amax(source));
        # print('np.amin(target) : ',np.amin(target));
        # print('np.amax(target) : ',np.amax(target));

        recon = []

        target_len = len(target);
        source_len = len(source);
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        iterations = 0;

        _stin = [];

        for epoch in range(epochs):

            n_batches = int(55000.0/batch_size);
            print("For batch_size : ",batch_size,' n_batches : ',n_batches);

            #n_batches = 50;
            for batch in range(n_batches):
                iterations += 1;
                #picks randomly training instances
                #sin = source[np.random.choice(len(target), batch_size)]
                sin = source[np.random.choice(source_len, batch_size)]
                tin = target[np.random.choice(target_len, batch_size)]
                sd = np.zeros((batch_size, 1)) #SOURCE IS Labelled 0
                td = np.ones((batch_size, 1)) #TARGET IS Labelled 1

                _stin = stin = np.concatenate((sin, tin), axis = 0)
                std = np.concatenate((sd, td), axis = 0)
                
                fd = {inpt:stin,domain:std};                

                ### SCHEDULE ###

                #learn domain discriminator
                _,_disc_loss = sess.run([disc_op,discriminator_loss],feed_dict=fd);
                
                #confuse domain discriminator
                _,_enc_inv_loss = sess.run([enc_inv_op,inv_encoder_loss],feed_dict=fd);

                #help domain discriminator
                _,_enc_var_loss = sess.run([enc_var_op,var_encoder_loss],feed_dict=fd);

                #regularize the encoders
                _,_recon_loss,recon,_merged_all = sess.run([recon_op,recon_loss,dec,merged_all],feed_dict=fd);

                if(iterations%20==0):
                    writer.add_summary(_merged_all,iterations);

                if(batch%100==0):
                    print("Batch #",batch," Done !");

            print('-'*80);
            print("Epoch #",epoch," completed !!");
            print('-'*80);

            if(epoch%5==0):
                #_stin = _stin.reverse();
                n = 5;
                random_source_batch = source[np.random.choice(source_len, n*n)];
                random_target_batch = target[np.random.choice(target_len, n*n)];
                
                source_fd = {inpt:random_source_batch};
                target_fd = {inpt:random_target_batch};

                recon_source = sess.run(dec_test ,feed_dict=source_fd);
                recon_target = sess.run(dec_test ,feed_dict=target_fd);
                
                reconstructed_source = np.empty((32*n,32*n));
                original_source = np.empty((32*n,32*n));
                reconstructed_target = np.empty((32*n,32*n));
                original_target = np.empty((32*n,32*n));
                for i in range(n):
                    for j in range(n):
                        original_source[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = random_source_batch[i*n+j].reshape([32, 32]);
                        reconstructed_source[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = recon_source[i*n+j].reshape([32, 32]);
                        original_target[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = random_target_batch[i*n+j].reshape([32, 32]);
                        reconstructed_target[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = recon_target[i*n+j].reshape([32, 32]);
                    
                print("Original_source Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original_source, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/orig-img-'+str(epoch)+'source.png');
                plt.close();

                print("Reconstructed_source Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed_source, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/recons-img-'+str(epoch)+'source.png');
                plt.close();

                print("Original_target Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original_target, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/orig-img-'+str(epoch)+'target.png');
                plt.close();

                print("Reconstructed_target Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed_target, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/recons-img-'+str(epoch)+'target.png');
                plt.close();
                
                # imsave("op/s_orig_"+str(epoch)+".jpg",_stin[0,:,:,0]);
                # imsave("op/s_rec_"+str(epoch)+".jpg", recon[0,:,:,0]);
                # imsave("op/t_orig_"+str(epoch)+".jpg",_stin[-1,:,:,0]);
                # imsave("op/t_rec_"+str(epoch)+".jpg", recon[-1,:,:,0]);
                saver.save(sess, checkpoint_path)

else:
    print("Inference mode activated :)");
    source = np.load("dataset/mnist.npy");
    source_labels = np.load("dataset/mnist_l.npy");

    target = np.load("dataset/svhn_test.npy");
    tgt_labels = np.load("dataset/svhn_l_test.npy");
    
    source_len = len(source);
    target_len = len(target);
    
    print('source.shape : ',source.shape);
    print('source_labels.shape : ',source_labels.shape);

    print('target.shape : ',target.shape);
    print('tgt_labels.shape : ',tgt_labels.shape);

    clf_checkpoint_path = './model_with_classifier'
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        #params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='encoder1');
        saver = tf.train.Saver(var_list=params);

        path_for_weights = checkpoint_path;

        try:
            saver.restore(sess, path_for_weights);
        except:
            #print("Previous weights not found of invariant encoder !"); 
            print("Previous weights not found of model"); 
            sys.exit(0);

        print ("Full model loaded successfully :)");

        for epoch in range(clf_train_epochs):
            
            # print('X.shape : ',target_X.shape);
            # print('Y.shape : ',target_Y.shape);
            
            # print('Y[0] : ',Y[0]);
            # print('class : ',np.argmax(Y[0]));

            # plt.title(np.argmax(Y[0]));
            # plt.imshow(X[0].reshape([32,32]), origin="upper",interpolation='nearest', cmap="gray");
            # plt.show();
            # plt.close();
            clf_batch_size = 128;
            n_batches = int(1.0*source_len/clf_batch_size);
            print('n_batches : ',n_batches,' for batch_size : ',clf_batch_size);

            _train_accuracy = 0.0;
            for batch in range(n_batches):
                random_indexes = np.random.choice(source_len, clf_batch_size);
                src_X = source[random_indexes];
                src_Y = source_labels[random_indexes];
                
                src_fd = {inpt:src_X,src_labels:src_Y};
                _,_src_clf_loss,_train_accuracy = sess.run([clf_op,clf_loss,train_accuracy],feed_dict=src_fd);
                

                if(batch%100==0):
                    print("Batch #",batch," Done!");

            
            print("Epoch #",epoch," Done with train accuracy : ,",_train_accuracy*100," !!");
            if(epoch%2==0):

                target_random_indexes = np.random.choice(target_len, clf_batch_size);

                target_X = target[target_random_indexes];
                target_Y = tgt_labels[target_random_indexes];
        
                target_fd = {inpt:target_X,target_labels:target_Y};
                
                _test_accuracy = sess.run(test_accuracy,feed_dict=target_fd);
                
            
                #_src_clf_loss is loss corresponding to last epoch
                print("Train accuracy : ",_train_accuracy*100," Test accuracy : ",_test_accuracy*100);
                saver.save(sess, clf_checkpoint_path)








        
        
