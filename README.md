# Railgun

[![The MIT License](https://img.shields.io/badge/license-MIT-orange.svg?style=flat-square)](http://opensource.org/licenses/MIT)


```
+++
  ____       _ _                   
 |  _ \ __ _(_) | __ _ _   _ _ __           _____                   ""',,. ,
 | |_) / _` | | |/ _` | | | | '_ \       __(     )============:: o''        ''>
 |  _ < (_| | | | (_| | |_| | | | | - -_|~~~~~~~~~~~\___  - - - -''.....""- -
 |_| \_\__,_|_|_|\__, |\__,_|_| |_| \\ \o-o_o-o_o-o_o-o_\ \\  \\  \\  \\  \\  \
                 |___/                - - - - - - - - - - - - - - - - - - - - -
```

Railgun is a task scheduler for CUDA C. We can overlap kernel executions for
high performance, but it is difficult to decide the order. The goal of this
project is to free you from considering it. Railgun determines the order instead
of you to execute your tasks for effective use of GPU resources.

## Examples
```
__global__ void matrix_add(int width, int height, double* A, double* B, double* C);

...

railgun_t *rg;
railgun_args *args;
int w, h;
double *hA, *hB, *hC;

...(set values into w, h, hA, hB and hC.)

rg = get_railgun();

// Task 1
args = rg->wrap_args("IIdd|d", w, 1, h, 1, hostA, w * h, hostB, w * h, hostC, w * h); 
rg->schedule((void*)matrix_add, args, dim3(1, 1), dim3(w, h));

// Task 2
args = rg->wrap_args("IIdd|d", w, 1, h, 1, hostA, w * h, hostB, w * h, hostC, w * h); 
rg->schedule((void*)matrix_add, args, dim3(1, 1), dim3(w, h));

// Task1 and Task 2 is executed concurrently.
rg->execute();

rg->reset_railgun();
```

## Build
If you have GPU on remote machine, please execute deploy.sh.

```
./deploy.sh [REMOTE_HOST] [REMOTE_DIR] [MAIN_FILE]
```

REMOTE_HOST:host which you execute the program. ex)foo@example.com or IP address

REMOTE_DIR:directory you deploy the program. ex) /home/yourname/projects

MAIN_FILE:file includes the main function

If you develop the program on a host has GPU, just execute make command.

## Contributing
Send a pull request to <http://github.com/rkoder/railgun>.
We welcome your great idea for this project. Feel free to
contribute your code to the project. 
You can use <http://github.com/rkoder/railgun/issues> for discussion. 

## See also

[rkoder/bheap](https://github.com/rkoder/bheap): general-purpose implementation of task queue in railgun

[Implementation and Evaluation of Data Transfer Scheduling in CUDA](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=180525&item_no=1&page_id=13&block_id=8): thesis (written in Japanese)

## License

MIT license (© 2019 Ryota Kota)
