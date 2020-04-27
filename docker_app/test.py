import urllib.request

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}


file_path="yolov3.weights"

with open(file_path, "wb") as output_file:
    with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
        length = int(response.info()["Content-Length"])
        counter = 0.0
        MEGABYTES = 2.0 ** 20.0


        #while True:
        data = response.read(8192)

        #if not data:
            #break

        counter += len(data)
        output_file.write(data)
