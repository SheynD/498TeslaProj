import org.apache.spark.rdd.RDD
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp._
import java.util.Properties
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.ling._
import Main.spark.implicits._
import org.apache.spark.util.collection._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.MinMaxScaler

object SentimentModel {
  //Regex statements for handling different patterns that appear in tweets, such as emojis or Twitter handles
  val emojiPattern = """[\p{So}+|\uFE0F]"""
  val handlePattern = """@[\w|_]+"""
  
  //Get the sentiment value for a labeled tweet in the dataset
  def computeSentimentLabeledTweet(tweetDS: Dataset[LabeledTweet]) = {
    tweetDS.mapPartitions{ part => 
	    val pipeline = createStandfordNLP
	    part.map(tweet => performSentiment(tweet, pipeline))
	  }
  }
  
   //Use NLP to tage the tweet and get a sentiment value for each tweet in a given day
   def computeSentimentDatedTweet(tweetDS: Dataset[DatedTweet]) = {
    tweetDS.mapPartitions{ part => 
	    val pipeline = createStandfordNLP
	    part.map(tweet => performSentiment(tweet, pipeline))
	  }
  }
  

  //Perform the sentiment analysis by normalizing text and analyzing each sentence
  def performSentiment(tweet : BaseTweet, pipeline : StanfordCoreNLP) = {
  
    //Remove emojis and handles from all tweets
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
    
    //Sum up the sentiment values and assign them to one of three cases
    val totalSentiment = sentiments.sum match {
      case negative if(negative < -3) => 0
      case positive if(positive > 1) => 2
      case neutral => 1
    }

    SentimentTweet(tweet.asInstanceOf[LabeledTweet].label, tweet.text, tweet.id, totalSentiment.toDouble)

	}
  //creates and initializes the Stanford NLP object
	def createStandfordNLP() : StanfordCoreNLP = {
	  val stanfordProps = new Properties()
    stanfordProps.put("annotators", "tokenize, ssplit, parse, sentiment")
    new StanfordCoreNLP(stanfordProps)
	}
	
  //Word2Vec algorithm for classifying tweets
  def computeWord2Vec(tweetDS: Dataset[LabeledTweet]) = {
	  val word2vec = new Word2Vec()
  	                    .setInputCol("wordsArray")
  	                    .setOutputCol("result")
                        .setVectorSize(50)
                        .setMinCount(0)
                        
    val tweetDSWithArray: Dataset[TempWord2VecTweet] = tweetDS.map{ labeledTweet => 
      val words = labeledTweet.text.replaceAll(handlePattern, "").split(" ").filterNot(_.isEmpty).filter(_.exists(c => c.isLetterOrDigit))
      TempWord2VecTweet(labeledTweet.label, labeledTweet.text, labeledTweet.id, words)
    }

	  val model = word2vec.fit(tweetDSWithArray)
	  val tweetDSWithWord2Vec = model.transform(tweetDSWithArray)
	  
	  //scale 0 to 1 for naive bayes 
	  val scaler = new MinMaxScaler()
                .setInputCol("result")
                .setOutputCol("features")
                
    val scalerModel = scaler.fit(tweetDSWithWord2Vec)
    (scalerModel.transform(tweetDSWithWord2Vec).as[LabeledWord2VecTweet], model)            
	}

}

//Create classes for different types of classified and unclassified tweets and use them to analyze the result
class BaseTweet(val text: String, val id: String) extends Serializable 
case class SentimentTweet(label: Double, override val text: String, override val id: String, sentiment: Double) extends BaseTweet(text, id)
case class LabeledWord2VecTweet(label: Double, override val text: String, override val id: String, features: Vector) extends BaseTweet(text, id)
case class TempWord2VecTweet(label: Double, text: String, id: String, wordsArray: Array[String])                   
case class Word2VecTweet(date: java.sql.Date, override val text: String, override val id: String, wordsArray: Array[String]) extends BaseTweet(text, id)  
case class DatedTweet(date: java.sql.Date, override val text: String, override val id: String) extends BaseTweet(text, id) 
case class SentimentTweetNoLabel(date: java.sql.Date, override val text: String, override val id: String, sentiment: Double) extends BaseTweet(text, id)
