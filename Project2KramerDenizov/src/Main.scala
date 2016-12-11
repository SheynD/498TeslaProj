import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

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
	  
	  tweetDS.show
	  
	  val tweetsWithSenitmentDS = SentimentModel.computeSentiment(tweetDS)
	  tweetsWithSenitmentDS.show
	  
	  //val tweetsWithFeatures = SentimentModel.word2Vec(tweetDS)
	  //tweetsWithFeatures.show
	  
	  val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("sentiment")
      .setMetricName("accuracy")
	  
    val accuracy = evaluator.evaluate(tweetsWithSenitmentDS)
    println(s"Accuracy: $accuracy")  
    
    val totalTime = (System.nanoTime - startTime)/1E9
    println(s"Time: $totalTime")
    
	  /*
	  val teslaDS: Dataset[Tweet] = Preprocessing.loadTeslaTweets	  
	  println(teslaDS.count)
    //Graph.plotTweetsPerDay(teslaDS, true)
	  
    val teslaStockPriceDS: Dataset[StockPrice] = Preprocessing.loadTeslaStockPrice
    //Graph.plotStockPerDay(teslaStockPriceDS, true)
    
    Graph.plotStockAndTweetsPerDay(teslaDS, teslaStockPriceDS)
  	*/
	}
}