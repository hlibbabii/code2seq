private AmazonKinesisClient makeNewKinesisClient() {
        AmazonKinesisClient client = new AmazonKinesisClient(getKinesisCredsProvider(), getClientConfiguration());
        LOG.info("Using " + getRegion().getName() + " region");
        client.setRegion(getRegion());       
        return client;
    }
