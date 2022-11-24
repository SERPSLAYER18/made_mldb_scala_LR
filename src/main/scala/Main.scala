

import breeze.linalg.{DenseMatrix, _}
import breeze.numerics._
import breeze.stats.regression.leastSquares

import java.io._

object Main extends App {
  val TRAIN_FILENAME = "train_data.csv"
  val TEST_FILENAME = "test_data.csv"
  val OUTPUT_FILENAME = "predictions.csv"


  var train = csvread(new File(TRAIN_FILENAME), ',')
  val target = train(::, (train.cols - 1))
  train = train.delete(train.cols - 1, Axis._1)

  var test = csvread(new File(TEST_FILENAME), ',')
  val y_true = test(::, (test.cols - 1))
  test = test.delete(test.cols - 1, Axis._1)

  val result = leastSquares(train, target)


  val y_pred = result.apply(test)
  val mape = sum(abs(y_pred - y_true) / y_true) / test.rows

  val concated = DenseMatrix.horzcat(test, new DenseMatrix(test.rows, 1, y_pred.toArray))
  println("MAE: " + mape * 100 + "%")

  csvwrite(new File(OUTPUT_FILENAME), concated, separator = ',')
  println("Predictions are saved into " + OUTPUT_FILENAME)
}