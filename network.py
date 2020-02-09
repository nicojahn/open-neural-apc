import tensorflow as tf

maximum_concat = 5
variance_scaling = 100.
bias_min, bias_max = -2,5

class network():
    def __init__(self, input_dims, hidden_dims, output_dims, layer, learning_rate, accuracy_error_slack, dropout_rate, gpu):
        
        self.model = dict()
        self.model['i_dim'] = input_dims
        self.model['h_dim'] = hidden_dims
        self.model['o_dim'] = output_dims
        self.model['h_layer'] = layer
        self.model['lr'] = learning_rate
        self.model['accuracy_slack'] = accuracy_error_slack
        self.model['dropout_rate'] = dropout_rate
        self.model['use_cudnn'] = gpu
    
    def create_model(self):
        # input aka. X
        with tf.variable_scope("inputs"):
            self.model['input'] = tf.placeholder(tf.float32,shape=[None,None,self.model['i_dim']])
            # mask x and save mask for later
            # detecting the length of the sequences (a mask that is 1, where a frame exists and 0 where the sequence has already ended)
            self.sequence_mask = tf.minimum(self.model['input'],0.)+1.
            self.model['x'] = tf.multiply(self.model['input'],self.sequence_mask)
        
        # labels aka y
        with tf.name_scope("Labels"):
            self.model['label'] = tf.placeholder(tf.float32,shape=[None,None,self.model['o_dim']])
            # masks the label
            mask = tf.minimum(self.model['label'],0.)+1.
            self.model['y'] = tf.multiply(self.model['label'],mask)
        
        # grep some dimentions and create slack variable
        with tf.name_scope("Utilities"):
            with tf.variable_scope("slack"):
                self.model['slack'] = tf.constant(dtype=tf.float32,value=self.model['accuracy_slack'])
            with tf.variable_scope("batch_size"):
                self.model['batch_size'] = tf.to_int32(tf.shape(self.model['x'])[0])
            with tf.variable_scope("sequence_length"):
                self.model['seq_length'] = tf.to_int32(tf.shape(self.model['x'])[1])
        
        self.createMasks()
        self.predict(self.model['x'])
        self.calculateLoss()
        self.calculateAccuracy()
        self.tensorboardEpochMetric()
        self.optimize()
        
    def createMasks(self):
        with tf.name_scope("masks"):
            with tf.variable_scope("boundingmask"):
                self.model['bound_mask'] = tf.placeholder(tf.float32, shape=[None,None,self.model['o_dim']])
            with tf.variable_scope("labelmask"):
                self.model['label_mask'] = tf.placeholder(tf.float32, shape=[None,None,self.model['o_dim']])
            
            def maskHelper(bound,iter):
                return tf.maximum(0.,tf.minimum(bound,iter+1)-iter)
            
            def createMask(bound):
                raw_masks = tf.stack([maskHelper(bound,idx) for idx in range(maximum_concat)],0)
                label_masks = raw_masks * tf.expand_dims(tf.transpose(self.model['y'],[1,0,2]),2)
                return tf.reduce_sum(label_masks, 0)
                
            with tf.variable_scope("upperboundmask"):
                ### all for upper bound mask
                self.upper_bound = createMask(self.model['bound_mask'])
                self.model['upper_bound'] = self.upper_bound
                
            with tf.variable_scope("lowerboundmask"):
                ### all for lower bound mask
                self.lower_bound = createMask(self.model['bound_mask']-1)
                self.model['lower_bound'] = self.lower_bound
            
            with tf.variable_scope("intermediatemask"):
                self.intermediate_mask = (1.-self.model['label_mask']) * tf.reduce_mean(self.sequence_mask,2,keepdims=True)

    def predict(self,input):
        ###########################NETWORK##############################
        dense_out = self.input(input)
        lstm_out = self.lstm(dense_out)
        prediction = self.output(lstm_out)
        ###########################NETWORK##############################
        self.model['prediction'] = prediction

    def calculateLoss(self):
        self.prediction = self.model['prediction']

        with tf.name_scope("Last-Error"):
            
            last_label = self.model['label_mask']*self.upper_bound
            last_prediction = self.model['label_mask']*self.prediction
            
            last_boolean_mask = tf.greater(tf.reduce_mean(self.model['label_mask'],axis=2),0.)
            
            with tf.name_scope("Last-Labels"):
                self.last_label = tf.boolean_mask(last_label,last_boolean_mask)
            with tf.name_scope("Last-Predicitions"):
                self.last_prediction = tf.boolean_mask(last_prediction,last_boolean_mask)

            self.model['last_pred'] = self.last_prediction
            self.model['last_label'] = self.last_label
            
            #subtract
            self.subtracted_pred_label = tf.subtract(last_prediction,last_label)

        with tf.name_scope("Loss"):
            inter_med = tf.maximum(0.,self.prediction-self.upper_bound) +\
                        tf.minimum(0.,self.prediction-self.lower_bound)
                        
            self.intermediate_only = self.intermediate_mask*inter_med
            self.mask = tf.add(self.intermediate_only,self.subtracted_pred_label)
            self.model['error'] = tf.reduce_mean(axis=0, input_tensor=tf.abs(self.mask),keepdims=True)
            with tf.variable_scope("loss"):
                self.model['loss'] = tf.reduce_mean(self.model['error'])

    def calculateAccuracy(self):
        #summary is created with the average over all batches
        #stats with reset
        with tf.name_scope("Accuracy"):
            #runde die letzten predictions auf bzw ab
            difference = tf.subtract(self.last_prediction,self.last_label)
            self.mini = tf.minimum(0.,difference+self.model['slack'])
            self.maxi = tf.maximum(0.,difference-self.model['slack'])
            self.model['error_with_slack'] = self.maxi+self.mini
            
            num_right = tf.math.equal(tf.zeros_like(self.model['error_with_slack']),self.model['error_with_slack'])
            num_elems = tf.math.equal(self.model['error_with_slack'],self.model['error_with_slack'])
            self.model['acc'] = tf.math.reduce_sum(tf.cast(num_right,dtype=tf.float32))/tf.math.reduce_sum(tf.cast(num_elems,dtype=tf.float32))
        
    def tensorboardEpochMetric(self):
        # epoch metrics
        with tf.name_scope("reset_acc"):
            self.model['epoch_acc'], self.model['epoch_acc_update'] = tf.metrics.mean(self.model['acc'],name="mean_acc")
            self.model['epoch_acc_scalar'] = tf.summary.scalar('Accuracy', self.model['epoch_acc'])
            locals = [v for v in tf.local_variables() if 'reset_acc/' in v.name]
            self.model['epoch_acc_reset'] = tf.variables_initializer(locals)
        
        with tf.name_scope("reset_loss"):
            self.model['epoch_loss'], self.model['epoch_loss_update'] = tf.metrics.mean(self.model['loss'],name="mean_cost")
            self.model['epoch_loss_scalar'] = tf.summary.scalar('Loss', self.model['epoch_loss'])
            locals = [v for v in tf.local_variables() if 'reset_loss/' in v.name]
            self.model['epoch_loss_reset'] = tf.variables_initializer(locals)

    def optimize(self):
        with tf.name_scope("Optimizer"):
            #Adam Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.model['lr'])
            with tf.variable_scope("Adam"):
                self.model['opt'] = optimizer.minimize(loss=self.model['error'])

    def input(self,input):
        initializer = tf.contrib.layers.variance_scaling_initializer(variance_scaling)
        # flatten input as if it would be only one sequence of self.model['i_dim'] images
        # reason: large matrix multiplication is faster
        input = tf.reshape(input,[-1,self.model['i_dim']])
        w_ih = tf.get_variable(name='W_in',shape=[self.model['i_dim'],self.model['h_dim']], initializer=initializer)
        b_ih = tf.get_variable('b_in',initializer=tf.random_uniform([self.model['h_dim']],bias_min, bias_max))
        dense_out = tf.nn.xw_plus_b(input,w_ih,b_ih)
        dense_out = tf.nn.leaky_relu(dense_out,name='activation')
        # reshape it into the old shape for lstm computations
        dense_out = tf.reshape(dense_out,[self.model['batch_size'],self.model['seq_length'],self.model['h_dim']])
        return dense_out
        
    def lstm(self,dense_out):
        if self.model['use_cudnn']:
            # cuddnn lstm (is time major, therefore transposing sequences is necessary)
            dense_out = tf.transpose(dense_out,[1,0,2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=self.model['h_layer'], dropout = self.model['dropout_rate'], num_units=self.model['h_dim'], dtype=tf.float32)
            lstm_out, states = lstm(dense_out,training = True)
            lstm_out = tf.transpose(lstm_out,[1,0,2])
        else:
            ###############THE CUDNN LSTM SHOULD BE DOWN HERE##################################
            with tf.variable_scope("cudnn_lstm"):
                single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.model['h_dim'])
                dropout_cell = lambda: tf.contrib.rnn.DropoutWrapper(single_cell(), output_keep_prob=1-self.model['dropout_rate'])
                cell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell() for _ in range(self.model['h_layer'])])
                initial_state = cell.zero_state(self.model['batch_size'], tf.float32)
                # Leave the scope arg unset.
                lstm_out, final_state = tf.nn.dynamic_rnn(cell, dense_out, initial_state=initial_state,dtype=tf.float32)
        return lstm_out
        
    def output(self,lstm_out):
        initializer = tf.contrib.layers.variance_scaling_initializer(variance_scaling)
        # make again a 2D - Vector from it (boost it)
        lstm_out = tf.reshape(lstm_out, [self.model['batch_size']*self.model['seq_length'],self.model['h_dim']])
        w_ho = tf.get_variable(name='W_out',shape=[self.model['h_dim'],self.model['o_dim']], initializer=initializer)
        b_ho = tf.get_variable('b_out',initializer=tf.random_uniform([self.model['o_dim']],bias_min, bias_max))
        prediction = tf.nn.xw_plus_b(lstm_out,w_ho,b_ho)
        prediction = tf.abs(prediction)
        # reshape it into the target shape
        prediction = tf.reshape(prediction,[self.model['batch_size'],self.model['seq_length'],self.model['o_dim']])
        return prediction
        
