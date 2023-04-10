import urllib.request
import json
import pyttsx3
import requests

response = requests.get("http://ip-api.com/json/103.220.210.215").json()

print(latitude = response['lat'])
print(longitude = response['lon'])


engine = pyttsx3.init()
# Your Bing Maps Key 
bingMapsKey = "ApFr1grOeMUw9DEV4sPm60bcgz1Ye6flK6FHfPqN97tDp0BsJVsf9uQxA1Myo2AF"

destination = "Sovabazar Raj Bari"

encodedDest = urllib.parse.quote(destination, safe='')

routeUrl = "http://dev.virtualearth.net/REST/V1/Routes/Driving?wp.0=" + str(latitude) + "," + str(longitude) + "&wp.1=" + encodedDest + "&key=" + bingMapsKey

request = urllib.request.Request(routeUrl)
response = urllib.request.urlopen(request)

r = response.read().decode(encoding="utf-8")
result = json.loads(r)

itineraryItems = result["resourceSets"][0]["resources"][0]["routeLegs"][0]["itineraryItems"]

for item in itineraryItems:
    engine.say(item["instruction"]["text"])
    engine.runAndWait()
