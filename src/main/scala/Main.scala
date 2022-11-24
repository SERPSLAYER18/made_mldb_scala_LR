

import breeze.linalg.{DenseMatrix, _}
import breeze.numerics._
import breeze.stats.regression.leastSquares

import java.io._

object Main extends App {
  val TRAIN_FILENAME = "train_data.csv"
  val TEST_FILENAME = "test_data.csv"
  val OUTPUT_FILENAME = "predictions.csv"
  val target = train(::, (train.cols - 1))
  val result = leastSquares(train, target)
  train = train.delete(train.cols - 1, Axis._1)
  val y_true = test(::, (test.cols - 1))
  val y_pred = result.apply(test)
  val mape = sum(abs(y_pred - y_true) / y_true) / test.rows
  test = test.delete(test.cols - 1, Axis._1)
  val concated = DenseMatrix.horzcat(test, new DenseMatrix(test.rows, 1, y_pred.toArray))
  var train = csvread(new File(TRAIN_FILENAME), ',')
  println("MAE: " + mape * 100 + "%")
  var test = csvread(new File(TEST_FILENAME), ',')
  println(concated)
  csvwrite(new File(OUTPUT_FILENAME), concated, separator = ',')
}