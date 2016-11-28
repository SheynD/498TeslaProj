import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

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
	  
	  val teslaDS: Dataset[Tweet] = Preprocessing.loadTeslaTweets	  
    //Graph.plotTweetsPerDay(teslaDS, true)
	  
    val teslaStockPriceDS: Dataset[StockPrice] = Preprocessing.loadTeslaStockPrice
    //Graph.plotStockPerDay(teslaStockPriceDS, true)
    
    Graph.plotStockAndTweetsPerDay(teslaDS, teslaStockPriceDS)
  }
}