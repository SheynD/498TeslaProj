import org.apache.spark.rdd.RDD
import Main.spark.implicits._
import java.text.SimpleDateFormat
import org.apache.spark.sql.types.TimestampType
import java.sql.Date
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.types._

//Preprocesses the tweets and files in order to perform data collections, label predictions, and classification
object Preprocessing {
  //The paths for the data files for tweets and stock prices
  val rawTeslaTweetPath = "data/tweets/teslaRaw/*"
  val teslaDSPath = "data/tweets/teslaDS"
  val rawTeslaStockPricePath = "data/stock/teslaStockHistory.csv"
  
  /*
   * Loads Tesla Tweets into a DataSet of Tweets
   * On first run, saves loaded Dataset to file for decreased runtime in later runs
   * Does not drop duplicates before saving to file because otherwise it causes a heap overflow 
   */
  def loadTeslaTweets(): Dataset[Tweet] = {
    try{
      Main.spark.read.load(teslaDSPath).as[Tweet].dropDuplicates("id") 
    }catch{
      case ex: Throwable => 
      
      //Convert the RDD to a tweet sequence using regex statements
      	def convertStringRDDToTweetSeq(rdd: RDD[String]): Seq[Tweet] = {
      			def splitLine(line: String): Array[String] = {
      					val semicolonsOutsideOfQuotesPattern = """;(?=(?:[^"]*"[^"]*")*[^"]*$)"""
      							line.split(semicolonsOutsideOfQuotesPattern)    
      			}
      			
      			val tweetRDD: RDD[Option[Tweet]] = rdd.filter(_ != "username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink") //remove headers
      					.map{line => 
        					val columns = splitLine(line)
        					try{
        						val rawTime = columns(1)
        						val sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm")
        						val id = columns(8).replace("\"","")
        						if(id.forall(_.isDigit)){
        							Option(Tweet(columns(0),new java.sql.Timestamp(sdf.parse(rawTime).getTime), new java.sql.Date(sdf.parse(rawTime).getTime),columns(2).toInt,columns(3).toInt,columns(4),columns(5),columns(6),columns(7),id,columns(9)))        						  
        						}else{
        						  None
        						}
        					}catch{
        					case ex : Throwable => /*println(ex);*/ None
        					}
      					}
      					tweetRDD.collect.flatten.toSeq
      	}
      	
      	val rawBadTweets: RDD[String] = Main.spark.read.text(rawTeslaTweetPath).as[String].rdd
      	rawBadTweets.cache
  			val goodTweetSeq: Seq[Tweet] = convertStringRDDToTweetSeq(rawBadTweets)
  			rawBadTweets.unpersist()
  			
  			//convert to DataSet[Tweet] and remove any duplicate tweets
  			val tweetDS: Dataset[Tweet] = Main.spark.sqlContext.createDataset(goodTweetSeq)

  			//save dataset for fast loading next time
  			tweetDS.write.mode(SaveMode.Overwrite).save(teslaDSPath) 
  			tweetDS.dropDuplicates("id") 
    }
  }
  
  
  //Load the stock price CSV file and select the columns of interest, and put them into 
  def loadTeslaStockPrice(): Dataset[StockPrice] = {
    val rawDF = Main.spark.read.option("header", "true").csv(rawTeslaStockPricePath)
    rawDF.select(rawDF("date").cast(DateType),
                 rawDF("open").cast(DoubleType),
                 rawDF("high").cast(DoubleType),
                 rawDF("low").cast(DoubleType),
                 rawDF("close").cast(DoubleType),
                 rawDF("volume").cast(LongType),
                 rawDF("adjClose").cast(DoubleType))
         .as[StockPrice]
  }
}

//Create a constant Tweet class template with the following attributes
final case class Tweet(username: String,
                       timestamp: java.sql.Timestamp,
                       date: java.sql.Date,
                       retweets: Int,
                       favorites: Int,
                       text: String,
                       geo: String,
                       mentions: String,
                       hashtags: String,
                       id: String,
                       permalink: String) extends Serializable 
                  
//Create a constant StockPrice class template with the following attributes
final case class StockPrice(date: java.sql.Date,
                            open: Double,
                            high: Double,
                            low: Double,
                            close: Double,
                            volume: Long,
                            adjClose: Double)          
                       
