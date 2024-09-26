@echo off

@set PATH_TO_JAR=C:\DevTools\PlantUML\plantuml.jar

java -Dfile.encoding=UTF-8 -jar %PATH_TO_JAR% %*
