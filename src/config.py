# License: MIT
# Author: Karl Stelzner


class SupairConfig:
    def __init__(self):
        # model config
        self.lstm_units = 256
        # native size of objects y^i
        self.patch_width = self.patch_height = 28
        # bounds for width of bounding box relative to native width
        self.min_obj_scale = 0.3
        self.max_obj_scale = 0.9
        # bounds for height of bounding box relative to width
        self.min_y_scale = 0.75
        self.max_y_scale = 1.25
        # number of inference steps (= N_max)
        self.num_steps = 3
        self.device = None  # specify tf device
        # choose between random SPN structure and Poon/Domingos structure
        self.random_structure = True
        self.background_model = True  # use learned background model
        # Parameter for penalty term in p(z), discouraging overlap
        self.overlap_beta = 120.0
        # Bounds on variance for leafs in object SPN
        self.obj_min_var = 0.12
        self.obj_max_var = 0.35
        # Bounds on variance for leafs in background SPN
        self.bg_min_var = 0.002
        self.bg_max_var = 0.12

        # learning config
        self.load_params = False
        self.save_params = True
        self.checkpoint_dir = 'checkpoints'
        self.batch_size = 256
        self.num_epochs = 10

        # data config
        self.dataset = 'MNIST'  # select dataset from 'MNIST', 'sprites', 'omniglot'
        self.data_path = './data'  # path to directory for loading and storing data
        self.scene_width = self.scene_height = 50
        self.channels = 1
        self.noise = False  # add Gaussian noise
        self.structured_noise = False  # add background grid

        # output config
        self.log_file = 'perf_log.csv'
        self.result_path = 'results/'
        # Measure and report count accuracy on test set
        self.get_test_acc = False
        self.visual = True  # Visualize training progress using visdom
        self.log_every = 100  # Output progress every log_every batches
