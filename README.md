# Railgun

[![Gitter](https://badges.gitter.im/rk0der/railgun.svg)](https://gitter.im/rk0der/railgun?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

```
+++
  ____       _ _                   
 |  _ \ __ _(_) | __ _ _   _ _ __           _____                   ""',,. ,
 | |_) / _` | | |/ _` | | | | '_ \       __( R.K.)============:: o''        ''>
 |  _ < (_| | | | (_| | |_| | | | | - -_|~~~~~~~~~~~\___  - - - -''.....""- - -
 |_| \_\__,_|_|_|\__, |\__,_|_| |_| \\ \o-o_o-o_o-o_o-o_\ \\  \\  \\  \\  \\  \\
                 |___/                - - - - - - - - - - - - - - - - - - - - - -
```

Railgun is a task scheduler for CUDA C. We can overlap kernel executions for high performance,
but it is difficult to decide the order. The goal of this project is to free you from considering it.
Railgun determines the order instead of you to execute your tasks for effective use of GPU resources.

## Examples
Under construction...

## Build
If you have GPU on remote machine, please execute deploy.sh.
```
./deploy.sh [REMOTE_HOST] [REMOTE_DIR]
```
REMOTE_HOST:host which you execute the program. GPU is needed. ex)foo@example.com or IP address  
REMOTE_DIR:directory you deploy the program. ex) /home/yourname/projects

If you develop the program on a host has GPU, just execute make command.

## Contributing
Send a pull request to <http://github.com/rk0der/railgun>.  We consider
you have granted non-exclusive right to your contributed code under
MIT license.  Use <http://github.com/rk0der/railgun/issues> for
discussion. You can also use gitter for more active conversation.

## License

MIT license (© 2016 Ryota Kota)
