import argparse


def main():
    parser = argparse.ArgumentParser(description='MISR ConvLSTM Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')

    args = parser.parse_args()

    if args.mode == 'train':
        from train import train_unified_model
        train_unified_model()
    elif args.mode == 'test':
        from test import test_unified_model
        test_unified_model()


if __name__ == "__main__":
    main()