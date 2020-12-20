from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    
    parser.add_argument("--data_dir", default='/home/zhl/AAAI2021/paper1/data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    
    parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")
    
    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 
    
    parser.add_argument("--bert_model", default="/home/zhl/pretrained_models/uncased_L-12_H-768_A-12", type=str)
    
    parser.add_argument("--max_seq_length", default=20, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--feat_dim", default=768, type=int)
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT")

    parser.add_argument("--save_model", action="store_true", help="save trained-model")

    parser.add_argument("--save_results", action="store_true", help="save test results")

    parser.add_argument("--pretrain", action="store_true", help="use pre-trained features")
    

    parser.add_argument("--dataset", default=None, type=str, required=True, 
                        help="The name of the task to train selected")
    
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True, help="The number of known classes")
    
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, required=True, help="The factor (magnification) of the number of clusters K")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--method", type=str, default='DeepAligned',help="which method to use")

    parser.add_argument("--labeled_ratio", default=0.1, type=float, help="The ratio of labeled samples in the training set")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop.") 

    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")

    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
    
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The learning rate for training.")  
    
    
    return parser
