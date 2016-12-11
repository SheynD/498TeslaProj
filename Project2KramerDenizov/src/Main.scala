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
	  
	  val tweetDS = Provided.loadAirLineTweets
	   
	  val result = SentimentModel.computeWord2Vec(tweetDS)
	  
	  val Array(training, testing) = result.randomSplit(Array(0.7, 0.3))
	  
	  val rf = new RandomForestClassifier()
            .setLabelCol("label")
            .setFeaturesCol("result")
            .setNumTrees(100)
	   
    val rfModel = rf.fit(training)
    val rfPredictions = rfModel.transform(testing)
    rfPredictions.show
    
    
    val layers = Array[Int](50, 100, 80, 100, 3)
    
    val trainer = new MultilayerPerceptronClassifier()
	                    .setLabelCol("label")
                      .setFeaturesCol("result")
                      .setLayers(layers)
                      .setBlockSize(128)
                      .setMaxIter(100)
                      
  val nnModel = trainer.fit(training)
  val nnResult = nnModel.transform(testing)
  
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
	  
    val accuracy = evaluator.evaluate(nnResult)
    println(s"accuracy $accuracy")

    
    
	  //val tweetsWithFeatures = SentimentModel.word2Vec(tweetDS)
	  //tweetsWithFeatures.show
	  
	   /*
	  val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("sentiment")
      .setMetricName("accuracy")
	  
    val accuracy = evaluator.evaluate(tweetsWithSenitmentDS)
    println(s"Accuracy: $accuracy")  
    
    val totalTime = (System.nanoTime - startTime)/1E9
    println(s"Time: $totalTime")
    */
	    
	  /*
	  val teslaDS: Dataset[Tweet] = Preprocessing.loadTeslaTweets	  
	  println(teslaDS.count)
    //Graph.plotTweetsPerDay(teslaDS, true)
	  
    val teslaStockPriceDS: Dataset[StockPrice] = Preprocessing.loadTeslaStockPrice
    //Graph.plotStockPerDay(teslaStockPriceDS, true)
     * 
    Graph.plotStockAndTweetsPerDay(teslaDS, teslaStockPriceDS)
  	*/
	}
}