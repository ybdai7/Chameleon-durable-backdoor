# Chameleon
Official implementation of **Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning** (https://arxiv.org/abs/2304.12961)

# Get started
To get started, you need to first install relevant packages using:
        pip install -r requirements.txt

Note: This repository largely follows the basic code structure of "How to backdoor federated learning" (https://github.com/ebagdasa/backdoor\_federated\_learning). Thus we use visdom to record and visualize experiment results.

After installing visdom, you need to initialize visdom using:
        python -m visdom.server -p port_number
The default port number is 8097 if not specify "-p port\_number". The visualization results can be found at localhost:port\_number.

If you are running using remote server, you need to run the following line at local terminal:
        ssh -L 1234:127.0.0.1:8097 username@server\_ip
Visualization result can be found at localhost:1234

# Run the code
To run the code, you can choose any command line from *commands.sh*. The results can then be found in visdom.

# Citation
We appreciate it if you would please cite the following paper if you found the repository useful for your work:
        @article{dai2023chameleon,
          title={Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning},
          author={Dai, Yanbo and Li, Songze},
          journal={arXiv preprint arXiv:2304.12961},
          year={2023}
        }
