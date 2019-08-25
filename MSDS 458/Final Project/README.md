# Final Project

Build and interpret a complex convolutional neural network on X-Ray image data. 

AWS GPU resources are used. The data is copied from S3 to an EBS volume in order to apply deep learning quickly.
-   Create IAM user with access to the bucket.
-   Configure AWS CLI for that user:  `aws configure`
-   Input the keys, region (us-east-1), and output type (text) as the inputs
Copy the files from S3 to the EBS volume: `aws s3 cp s3://<bucket name> <destination path> --recursive`