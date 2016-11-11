package clmet

import java.io.File
import scala.io.Source

object Common {
  def intOrNone(str: String): Option[Int] = {
    try {
      Some(str.toInt)
    } catch {
      case ex: Throwable => None
    }
  }

  def strOrNone(str: String): Option[String] = {
    if (str.trim.isEmpty) {
      None
    } else {
      Some(str)
    }
  }
}
