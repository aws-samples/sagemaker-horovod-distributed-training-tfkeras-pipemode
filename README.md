## Multi-GPU and distributed training using Horovod in Amazon SageMaker Pipe mode

This is a tutorial on how to run multi-GPU training on a single instance on [Amazon SageMaker](https://aws.amazon.com/sagemaker/), and then will move to efficient multi-GPU and multi-node distributed training on Amazon SageMaker. 



This example has several training examples with different configurations as follows:

- Training Tensorflow/Keras on local machine
- Running a training job on separate training instance(s) with File Mode input
- Running a training job on separate training instance(s) with Pipe Mode input
- Running a distributed training job with Horovod with File Mode input
- Running a distributed training job with Horovod with Pipe Mode input



This example extends this Amazon SageMaker example:

https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/keras_script_mode_pipe_mode_horovod/tensorflow_keras_CIFAR10.ipynb



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

