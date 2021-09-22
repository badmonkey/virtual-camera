

### v4l2loopback
The v4l2loopback kernel module can be installed through the package manager of your Linux
distribution or compiled from source following the instructions in the
[v4l2loopback github repository](https://github.com/umlaeute/v4l2loopback).

Once installed, the module needs to be loaded. This can be done manually for the current session by
running

    $ sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=2 card_label="fake-cam"
which will create a virtual video device `/dev/video2`, however, this will no persist past reboot.
(Note that the `exclusive_caps=1` option is required for programs such as Zoom and Chrome).

To create the virtual video device on startup, run the `v4l2loopback-install.sh` script to create
`/etc/modules-load.d/v4l2loopback.conf` to load the module and
`/etc/modprobe.d/linux-fake-background.conf` to configure the module.

The camera will appear as `fake-cam` in your video source list.

If you get an error like
```
OSError: [Errno 22] Invalid argument
```

when opening the webcam from Python, please try the latest version of v4l2loopback from the its
[Github repository](https://github.com/umlaeute/v4l2loopback), as the version from your package
manager may be too old.
