import java.util.Calendar
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import Main.spark.implicits._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.isnull
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import SentimentModel._
import org.apache.spark.ml.feature.{Word2VecModel, Word2Vec}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.Row
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp._
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.ling._
import java.util.Properties
import org.apache.spark.sql.SaveMode

//The actual model for using on the Tesla tweet and stock price datasets, with cross validation
object TeslaModel {
  
  //The constant number of max tweets to use from a day in training/testing the classifier
  val MAX_TWEETS_PER_DAY = 200
  
  val teslaDS: Dataset[Tweet] = Preprocessing.loadTeslaTweets	 
 
  //Tweet features to customize for the model (for instance, don't use Saturday and Sunday because they're not trading days)
  def createTweetOtherFeatures() = {
    
    val isWeekDay: (java.sql.Date => Boolean) = (date: java.sql.Date) => {
      val calendar: Calendar = java.util.Calendar.getInstance(); //needs to be in this loop otherwise indexoutofbounds exception
      calendar.setTime(date)
      calendar.get(Calendar.DAY_OF_WEEK) match{
        case Calendar.SATURDAY => false
        case Calendar.SUNDAY => false
        case _ => true
      }
    }
    //Used for weekdays (trading days)
    val isWeekDayUdf = udf(isWeekDay)
    
    //filter out weekends and group by date
    val teslaDSGrouped = teslaDS.filter(isWeekDayUdf('date)).groupBy('date)
    
    //Group the tweet counts and sum the retweets
    val teslaDSTweetCount = teslaDSGrouped.count().withColumnRenamed("count", "tweetCount")
    val teslaDSSumFavoritesRetweets = teslaDSGrouped.sum().withColumnRenamed("sum(retweets)", "sumRetweets").withColumnRenamed("sum(favorites)", "sumFavorites")
    
    //date|tweetCount|sumRetweets|sumFavorites
    val teslaDSFeatures = teslaDSTweetCount.join(teslaDSSumFavoritesRetweets, "date")
    teslaDSFeatures
  }
  
  //Label the days of the week for the stock price value and get the day from the date shown in the dataset
  def createLabeledPriceDays() = {
    val teslaStockPriceDS = Preprocessing.loadTeslaStockPrice.select('date, 'close).withColumnRenamed("close", "price")
       
    def shiftDaysUdf = udf[java.sql.Date, java.sql.Date](date => {
      val calendar: Calendar = java.util.Calendar.getInstance(); //needs to be in this loop otherwise indexoutofbounds exception
      calendar.setTime(date)
      calendar.get(Calendar.DAY_OF_WEEK) match{
        case Calendar.MONDAY => {
          calendar.add(Calendar.DAY_OF_MONTH, -3) //bring back to friday
          new java.sql.Date(calendar.getTimeInMillis)
        }
        case _ => {
          calendar.add(Calendar.DAY_OF_MONTH, -1) //minus 1 day
          new java.sql.Date(calendar.getTimeInMillis)
        }
      }
    })
    
    val teslaStockPriceShiftedDS = teslaStockPriceDS.withColumn("date", shiftDaysUdf('date)).withColumnRenamed("price","nextDayPrice")
    
    val stockBothDS = teslaStockPriceDS.join(teslaStockPriceShiftedDS, "date")
    //stockBothDS.show
    
    val labelPriceDelta: (Double, Double) => Double = (price: Double, nextDayPrice: Double) => if(nextDayPrice - price > 0) 1.0 else 0.0
    val labelPriceDeltaUdf = udf(labelPriceDelta)
    
    stockBothDS.withColumn("label", labelPriceDeltaUdf('price, 'nextDayPrice))
  }
  
  //Create the features to be used by the model
  def createAllFeatures() = {
    //Path to file that gives features to be used
    val path = "data/savedFeatures/featuresDS"
    
    //Try reading the file and if exception, create other features manually
    try{
      Main.spark.read.load(path)
    }catch{
      case ex: Throwable => 

      val other = createTweetOtherFeatures
      
      println("other count:" + other.count)
      
      val otherFeatures = other.join(createLabeledPriceDays, "date").sort('date)
      
      println("other features count:" + otherFeatures.count)
      
      val dateTextIDSentiment = computeSentimentByNLP
      val sentimentDS = dateTextIDSentiment.groupBy('date).sum("sentiment").withColumnRenamed("sum(sentiment)", "totalSentiment")
      
      println("sentiment count:" + sentimentDS.count)
        
      val allFeatures = otherFeatures.join(sentimentDS, "date")
      allFeatures.write.mode(SaveMode.Overwrite).save(path) 
      allFeatures
    }
  }
  
 //Use NLP to classify the tweets and get the sentiment value
  def computeSentimentByNLP() = {
    val teslaGroupedRDD = teslaDS.rdd.groupBy(_.date)
 
    //get n tweets from each day
    val teslaRandomDS: Dataset[Tweet] = teslaGroupedRDD.map{ case (date, iterator) =>
      val randomTweets = scala.util.Random.shuffle(iterator.toSeq).take(MAX_TWEETS_PER_DAY) //select n random tweets from a day
      randomTweets
    }.flatMap(identity).toDS().as[Tweet]
    
    val teslaDatedTweets: Dataset[DatedTweet] = teslaRandomDS.map(tweet => DatedTweet(tweet.date, tweet.text, tweet.id))
    
    println("total tweets count:" + teslaDatedTweets.count)
    
    computeSentimentDatedTweet(teslaDatedTweets)
  }
  
  //Use Word2Vec to classify the tweets and get the sentiment value
  def computeSentimentByWord2Vec(word2VecModel: Word2VecModel) = {
    val teslaGroupedRDD = teslaDS.rdd.groupBy(_.date)
 
    //get n tweets from each day
    val teslaRandomDS: Dataset[Tweet] = teslaGroupedRDD.map{ case (date, iterator) =>
      val randomTweets = scala.util.Random.shuffle(iterator.toSeq).take(MAX_TWEETS_PER_DAY) //select n random tweets from a day
      randomTweets
    }.flatMap(identity).toDS().as[Tweet]
    
    //fit the word2vec model
    val teslaWord2Vec = convertTweetsToWord2Vec(teslaRandomDS, word2VecModel)
    
    teslaWord2Vec
  }
  
  //Convert the tweet text to a Word2Vec vector using filters and model transformations
  def convertTweetsToWord2Vec(tweets: Dataset[Tweet], word2VecModel: Word2VecModel) = {
    val teslaWordsArrayDS = tweets.map{ tweet => 
      val words = tweet.text.replaceAll(handlePattern, "").split(" ").filterNot(_.isEmpty).filter(_.exists(c => c.isLetterOrDigit))
      Word2VecTweet(tweet.date, tweet.text, tweet.id, words)
    } 
    
    val word2VecTweets = word2VecModel.transform(teslaWordsArrayDS)   
    
    //MinMaxScale the features and results of the labels
    val scaler = new MinMaxScaler()
                .setInputCol("result")
                .setOutputCol("features")
                
    val scalerModel = scaler.fit(word2VecTweets)
    scalerModel.transform(word2VecTweets).as[Word2VecTweet]   
  }
  
  //Run the different models (Random Forest, Multilayerperceptron, Naive Bayes) as an ensemble model
  def runModels(tweets: Dataset[Word2VecTweet], rfModel: RandomForestClassificationModel, nnModel: MultilayerPerceptronClassificationModel, nbModel: NaiveBayesModel) = {
    
    //ensemble                  
	  val rfPredictions = rfModel.transform(tweets)
	                             .select('date, 'id, 'text, 'prediction)
	                             .withColumnRenamed("prediction", "rfPrediction")
	  
	  val nnPredictions = nnModel.transform(tweets)
	                             .select('date, 'id, 'text, 'prediction)
	                             .withColumnRenamed("prediction", "nnPrediction") 
	  
	  val nbPredictions = nbModel.transform(tweets) 
	                             .select('date, 'id, 'text, 'prediction)
	                             .withColumnRenamed("prediction", "nbPrediction")
	  rfPredictions.show
	  val allPredictions = rfPredictions.join(nnPredictions, Seq("date","id","text")).join(nbPredictions, Seq("date","id","text"))
	  
	  //Vote on the label using ranked voting system from predictions of ensemble methods
	  def rankedVoting: (Double, Double, Double) => Double = (pred1: Double, pred2: Double, pred3: Double) => { 
	    val votes = Array(pred1, pred2, pred3)
	    val groupedVotes = votes.groupBy(identity) //group by votes
	                            .mapValues(_.length) //count votes
	    if(groupedVotes.keys.size == 3){
	      1.0 //neutral, they all disagree
	    }else{
	      groupedVotes.maxBy(_._2)._1 //max of vote count and grab label
	    }
	  }          
	  
	  val rankedVotingUDF = udf(rankedVoting)
	  
	  val ensomblePredictions = allPredictions.withColumn("prediction", rankedVotingUDF($"rfPrediction", $"nnPrediction", $"nbPrediction"))
	  ensomblePredictions.show
  }
  
  def computeSentimentDatedTweet(tweetDS: Dataset[DatedTweet]) = {
    tweetDS.mapPartitions{ part => 
	    val pipeline = createStandfordNLP
	    part.map(tweet => performSentiment(tweet, pipeline))
	  }
  }
  
  //Perform the sentiment analysis by removing any emojis or handles and using the Stanford NLP to tag and process tweets
  def performSentiment(tweet : DatedTweet, pipeline : StanfordCoreNLP) = {
  
    val text = tweet.text.replaceAll(emojiPattern, "")
                         .replaceAll(handlePattern, "")

	  val document = new Annotation(text)
	  pipeline.annotate(document)
	  
	  val sentences = document.get(classOf[SentencesAnnotation])
	
	  val sentiments = for(sentence <- sentences; token <- sentence.get(classOf[TokensAnnotation])) yield {
	    val tree = sentence.get(classOf[SentimentAnnotatedTree]);
      val sentiment = RNNCoreAnnotations.getPredictedClass(tree) match {
        case 0 => -2 //very negative
        case 1 => -1 //negative
        case 2 => 0 //neutral
        case 3 => 1 //positive
        case 4 => 2 //very positive
      }
      sentiment
	  }
    //Compute the total sentiment of each day's tweets
    val totalSentiment = sentiments.sum match {
      case negative if(negative < -3) => -1
      case positive if(positive > 1) => 1
      case neutral => 0
    }

   SentimentTweetNoLabel(tweet.date, tweet.text, tweet.id, totalSentiment.toDouble)
	}
  //creates and initializes the Stanford NLP object
	def createStandfordNLP() : StanfordCoreNLP = {
	  val stanfordProps = new Properties()
    stanfordProps.put("annotators", "tokenize, ssplit, parse, sentiment")
    new StanfordCoreNLP(stanfordProps)
	}
}
