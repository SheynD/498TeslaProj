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
	  
	  println("Num negative: " + tweetDS.filter('label === 0.0).count)
	  println("Num neutral: " + tweetDS.filter('label === 1.0).count)
	  println("Num positive: " + tweetDS.filter('label === 2.0).count)
	  
	  /* 
	  val result = SentimentModel.computeWord2Vec(tweetDS)
	  
	  val Array(training, testing) = result.randomSplit(Array(0.7, 0.3))
	  
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
    
	  //ensemble                  
	  val rfPredictions = rfModel.transform(testing)
	                             .select('label, 'id, 'text, 'prediction)
	                             .withColumnRenamed("prediction", "rfPrediction")
	  
	  val nnPredictions = nnModel.transform(testing)
	                             .select('label, 'id, 'text, 'prediction)
	                             .withColumnRenamed("prediction", "nnPrediction") 
	  
	  val nbPredictions = nbModel.transform(testing) 
	                             .select('label, 'id, 'text, 'prediction)
	                             .withColumnRenamed("prediction", "nbPrediction")
	  
	  val allPredictions = rfPredictions.join(nnPredictions, Seq("label","id","text")).join(nbPredictions, Seq("label","id","text"))
	                             
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
	  
	  ensomblePredictions.sort('label desc).show(200)
	  
	  val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
	  
    val accuracy = evaluator.evaluate(ensomblePredictions)
    println(s"accuracy $accuracy")

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