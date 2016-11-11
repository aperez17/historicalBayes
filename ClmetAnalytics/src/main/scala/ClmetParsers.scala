package clmet

import scala.xml._

object ClmetParser {
  def nodeSeqToString(node: NodeSeq, attr: String): Option[String] = {
    try {
      Some((node \ attr).text)
    } catch {
      case _: Throwable => None
    }
  }
  def nodeSeqToInt(node: NodeSeq, attr: String): Option[Int] = {
    try {
      Some((node \ attr).text.toInt)
    } catch {
      case _: Throwable => None
    }
  }

  def parseClmetFile(xml: Elem): Vector[ClmetAnalysis] = {
    (for {
      header <- xml \\ "header"
      id <- header \\ "id"
      period <- header \\ "period"
      text <- header \\ "text"
    } yield {
      val idTxt = id.text
      val periodTxt = period.text
      val paragraphText = (
        for {
          paragraph <- text \\ "p"
        } yield {
            paragraph.text
        }
      ).toVector
      ClmetAnalysis(idTxt, periodTxt, paragraphText)
   }).toVector
  }

  def parseXml(xml: Elem)  = {
    parseClmetFile(xml)
  }
}
