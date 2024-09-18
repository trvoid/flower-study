@echo off

@set PATH_TO_JAR=C:\DevTools\PlantUML\plantuml-1.2023.6.jar

java -Dfile.encoding=UTF-8 -jar %PATH_TO_JAR% %1
