import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.Dataset

object Main {

  //initialize spark
	val spark = SparkSession.builder
              .master("local[6]")
              .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
              .getOrCreate()
  import spark.implicits._  
  
  //turn off spark info
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)
  
  //entry point    
	def main(args: Array[String]) : Unit = {
	 
	  val startTime = System.nanoTime
	  /*
	  val tweetDS = Provided.loadAirLineTweets
	  
	  println("Num negative: " + tweetDS.filter('label === 0.0).count)
	  println("Num neutral: " + tweetDS.filter('label === 1.0).count)
	  println("Num positive: " + tweetDS.filter('label === 2.0).count)
	  
	  val (training, word2vecModel) = SentimentModel.computeWord2Vec(tweetDS)
	   
	  //random forest
	  val rf = new RandomForestClassifier()
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setNumTrees(100)
	   
    val rfModel = rf.fit(training)

    //perceptron
    val layers = Array[Int](50, 100, 80, 100, 3)
    
    val trainer = new MultilayerPerceptronClassifier()
	                    .setLabelCol("label")
                      .setFeaturesCol("features")
                      .setLayers(layers)
                      .setBlockSize(128)
                      .setMaxIter(100)
                      
    val nnModel = trainer.fit(training)
    
    //naive bayes
    val nbModel = new NaiveBayes()
	                   .fit(training)
    
	  val teslaWithWord2Vec = TeslaModel.computeSentimentByWord2Vec(word2vecModel)
	  TeslaModel.runModels(teslaWithWord2Vec, rfModel, nnModel, nbModel)
	  
	  /*
	  val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
	  
    val accuracy = evaluator.evaluate(ensomblePredictions)
    println(s"accuracy $accuracy")

    * 
    * 
    * 
    */
    */
    TeslaModel.createAllFeatures.show
    
   
	  
    val totalTime = (System.nanoTime - startTime)/1E9
    println(s"Time: $totalTime")
    
    
	

	}
}