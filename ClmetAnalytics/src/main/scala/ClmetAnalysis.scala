package clmet

import clmet.Common._
import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.xml.XML

trait OutputAnalysis{
  def writeXmlToFile(outputPath: String, xml: ClmetAnalysis): Unit
}

case class PeriodOutputAnalysis() extends OutputAnalysis {
  def writeXmlToFile(outputPath: String, xml: ClmetAnalysis) = {
    val file = new File(outputPath + s"/${xml.period}/${xml.id}.txt")
    val printWriter = new PrintWriter(file)
    for {
      paragraph <- xml.text
    } yield {
      printWriter.println(paragraph)
    }
    printWriter.close()
  }
}

object DecadeOutputAnalysis {
  val FivePeriods = Vector((1710, 1752), (1752, 1794), (1794, 1836), (1836, 1878), (1878, 1921))
  val SevenPeriods = Vector((1710,1740),(1740,1770),(1770,1800),(1800,1830),(1830,1860),(1860,1890),(1890,1921))
}

case class DecadeOutputAnalysis(periods: Vector[(Int, Int)]) extends OutputAnalysis {
  
  def periodFromDecade(decade: Int) = {
    def isBetweenPeriod(periodRange: (Int,Int), decade: Int): Boolean = {
      val start = periodRange._1
      val end = periodRange._2
      start <= decade && decade < end
    }
    val output = periods.find(periodRange => isBetweenPeriod(periodRange,decade))
    if (output.isEmpty){
      println("2", decade)
    }
    output
  }
  
  def writeXmlToFile(outputPath: String, xml: ClmetAnalysis) = {
    val d = xml.decade
    if (d.isEmpty) {
      println("1", d)
    }
    for {
      decade <- xml.decade
      (start, end) <- periodFromDecade(decade)
    } yield {
      val file = new File(outputPath + s"/${start}-${end}/${xml.id}.txt")
      val printWriter = new PrintWriter(file)
      for {
        paragraph <- xml.text
      } yield {
        printWriter.println(paragraph)
      }
      printWriter.close()
    }
  }
}

case class ClmetAnalyzer(outputAnalysisType: OutputAnalysis) {

  def parseFiles(files: List[File]): Vector[ClmetAnalysis] = {
    files.flatMap(file => parseXmlFromFile(file)).toVector
  }

  def getAllFilesInDirectory(pathToDir: String): List[File] = {
    val d = new File(pathToDir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }
  
  def normalizeXml(xml: Vector[String]) = {
    val regex = "<page[ 0-9A-Za-z\"=]*/>|[&\";]".r
    val s = "<header>\n" + xml.reduceOption((a,b) => s"$a\n$b").getOrElse("") + "\n</header>"
    regex.replaceAllIn(s, "")
  }
  
  def parseXmlFromFile(file: File): Vector[ClmetAnalysis] = {
    val lines =  Source.fromFile(file).getLines.toVector
    val normalizedLines = normalizeXml(lines)
    val xml = XML.loadString(normalizedLines)
    ClmetParser.parseXml(xml)
  }
  
  def parseAllXmlFromDir(pathToDir: String): Vector[ClmetAnalysis] = {
    val files = getAllFilesInDirectory(pathToDir)
    parseFiles(files)
  }
  
  def writeXmlsToDir(pathToDir: String, xmls: Vector[ClmetAnalysis]) = {
    xmls.map(xml => outputAnalysisType.writeXmlToFile(pathToDir, xml))
  }
}

case class ClmetAnalysis(
    id: String,
    year: Option[Int] = None,
    decade: Option[Int] = None,
    period: Option[String] = None,
    text: Vector[String]
)
