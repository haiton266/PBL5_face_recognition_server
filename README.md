# PBL5 face recognition server

In this project for PBL5 at my university, this server will connect to ESP32-cam to auto-unlock the door.

Run server AI by this command:
```
uvicorn main:app --host 0.0.0.0 --port 5001 --reload
```

Some note if you deploy on a GPU server, you must install PyTorch-gpu with your cuda in this link: https://pytorch.org/get-started/locally/


After that, you can install other libraries regularly.