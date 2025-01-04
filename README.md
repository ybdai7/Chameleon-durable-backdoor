# Chameleon
Official implementation of **Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning** (https://proceedings.mlr.press/v202/dai23a)

# News
- [2024.09.27] Another repo of ours (https://github.com/ybdai7/Backdoor-indicator-defense), which has a more clear code structure and focuses on backdoor detection, also implements Chameleon (participants/clients/ChameleonMaliciousClient.py). You may want to refer to that repo if needed. Remeber to pay attention to the difference between yaml files. 

# Get started
To get started, you need to first install relevant packages using:
  
    pip install -r requirements.txt

Note: This repository largely follows the basic code structure of "How to backdoor federated learning" (https://github.com/ebagdasa/backdoor\_federated\_learning). Thus we use visdom to record and visualize experiment results.

After installing visdom, you need to initialize visdom using:

    python -m visdom.server -p port_number

The default port number is 8097 if not specify "-p port\_number". The visualization results can be found at localhost:port\_number.

If you are running using remote server, you need to run the following line at local terminal:

    ssh -L 1234:127.0.0.1:8097 username@server_ip

Visualization result can be found at localhost:1234

# Run the code
To run the code, you can choose any command line from *commands.sh*. The results can then be found in visdom.

For the edge-case datasets, you can acquire them following the instructions of https://github.com/ksreenivasan/OOD_Federated_Learning, which is the official repo of the Yes-you-can-really-backdoor-FL paper.

# Citation
We appreciate it if you would please cite the following paper if you found the repository useful for your work:


    @InProceedings{pmlr-v202-dai23a,
    title = 	 {Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning},
    author =       {Dai, Yanbo and Li, Songze},
    booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
    pages = 	 {6712--6725},
    year = 	 {2023},
    editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
    volume = 	 {202},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {23--29 Jul},
    publisher =    {PMLR},
    pdf = 	 {https://proceedings.mlr.press/v202/dai23a/dai23a.pdf},
    url = 	 {https://proceedings.mlr.press/v202/dai23a.html},
    abstract = 	 {In a federated learning (FL) system, distributed clients upload their local models to a central server to aggregate into a global model. Malicious clients may plant backdoors into the global model through uploading poisoned local models, causing images with specific patterns to be misclassified into some target labels. Backdoors planted by current attacks are not durable, and vanish quickly once the attackers stop model poisoning. In this paper, we investigate the connection between the durability of FL backdoors and the relationships between benign images and poisoned images (i.e., the images whose labels are flipped to the target label during local training). Specifically, benign images with the original and the target labels of the poisoned images are found to have key effects on backdoor durability. Consequently, we propose a novel attack, Chameleon, which utilizes contrastive learning to further amplify such effects towards a more durable backdoor. Extensive experiments demonstrate that Chameleon significantly extends the backdoor lifespan over baselines by $1.2\times \sim 4\times$, for a wide range of image datasets, backdoor types, and model architectures.}
    }

