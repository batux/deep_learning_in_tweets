package com.deeplearning4j.demo.word2vec;

import java.io.File;
import java.util.Collection;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 *
 * Batuhan Duzgun - Yeditepe University
 *
 * Reference-1: https://deeplearning4j.org
 * Reference-2: https://github.com/deeplearning4j/dl4j-examples
 */

public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static final String DATA_PATH = "/Users/batux/Documents/data/"; 
    //FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
    
    @SuppressWarnings("deprecation")
	public static void main(String[] args) throws Exception {

    	String filePath = "/Users/batux/Documents/raw_sentences.txt";
    	
        log.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(filePath);
        
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        WordVectorSerializer.writeWordVectors(vec, "/Users/batux/Documents/raw_sentences_output.txt");

        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println("10 Words closest to 'day': " + lst);
        
        
        
        
        int batchSize = 264;
        int vectorSize = 100;
        int nEpochs = 50;
        int truncateReviewsToLength = 2; 
        
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(nEpochs)
                .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(2e-2)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                    .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

      
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File("/Users/batux/Documents/raw_sentences_output.txt"));
        TweetDataIterator train = new TweetDataIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);

        
        org.deeplearning4j.ui.api.UIServer uiServer = org.deeplearning4j.ui.api.UIServer.getInstance();
		
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

		net.fit(train);
		
    }
}
