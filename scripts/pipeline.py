import argparse
from train import train
from sample import sample
import zero
import lib
import torch


def main():
    parser = argparse.ArgumentParser(
                    prog='MLDM',
                    description='Oversampling for multilabel datasets using diffusion models',
                    epilog='If you have any problems, please refer to the documentation')
    parser.add_argument('--config', metavar='FILE')

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)

    if 'num_threads' in raw_config:
        torch.set_num_threads(raw_config['num_threads'])
    else:
        torch.set_num_threads(1)

    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timer = zero.Timer()
    timer.run()

    train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        real_data_path=raw_config['real_data_path'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        device=device,
        strategy=raw_config['sample']['strategy'],
        label_percentage=raw_config['sample']['label_percentage']
    )
    sample(
        sample_percentage=raw_config['sample']['sample_percentage'],
        batch_size=raw_config['sample']['batch_size'],
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        real_data_path=raw_config['real_data_path'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        device=device,
        seed=raw_config['sample'].get('seed', 0),
        strategy=raw_config['sample']['strategy'],
        label_percentage=raw_config['sample']['label_percentage'],
        max_iter=raw_config['sample']['max_iterations'],
        output_file=raw_config['output_file'],
        quantile=raw_config['train']['T']['normalization']=="quantile"
    )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()