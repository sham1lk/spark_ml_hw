import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.made.DistributedLinearRegression
import org.apache.spark.sql.Row

object SimpleApp {

  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("DistributedLinearRegression")
      .config("spark.master", "local[4]")
      .getOrCreate()

    import spark.implicits._

    val numSamples = 1000
    val numFeatures = 3
    val data = (0 until numSamples).map { _ =>
      val features = Array.fill(numFeatures)(math.random)
      val noise = 0.1 * math.random
      val label = 1.5 * features(0) + 0.3 * features(1) - 0.7 * features(2) + noise
      (label, Vectors.dense(features))
    }.toDF("label", "features")

    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))

    val lr = new DistributedLinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
    val lrModel = lr.fit(trainData)

    val predictions = lrModel.transform(testData)

    val MSE = predictions.selectExpr("pow(prediction - label, 2) as squaredError").agg("squaredError" -> "avg").head().getDouble(0)
    println(s"Mean Squared Error: $MSE")
    val coefficients: Array[Double] = lrModel.weights.toArray

    println("Coefficients: ")
    coefficients.foreach(println)

    spark.stop()
  }
}