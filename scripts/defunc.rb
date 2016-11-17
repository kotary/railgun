#!/usr/bin/env ruby

# Descartes Exponentiation
def desc_exp(arr, n)
  if n > 1
    arr.product(desc_exp(arr, n - 1)).map {|x| x.flatten}
  else
    arr.map {|x| [x]}
  end
end

def gen_funcptr_name(fmt)
  fmt.map {|s| s[-1] == '*' ? s[0] : s[0].upcase}.join + '_f'
end

def gen_funcptr_def(type, fpname, params)
  "typedef #{type} (*#{fpname})(#{params});"
end

def gen_args(fmt)
  i = 0
  fmt
    .map {|s| s[-1] == '*' ? "#{s[0]}p" : s[0]}
    .map.with_index {|s,i| "mem[#{i}].#{s}"}.join(',')
end

def gen_execute_kernel(fpname, args)
  fmt_s = fpname[0..-3]
  "if (!strcmp(fmt, \"#{fmt_s}\")) ((#{fpname})task->f)<<<task->blocks, task->threads, 0, *strm>>>(#{args});"
end

n = ARGV[0].to_i
lines0 = []
lines1 = []

types = ['char', 'int', 'float', 'double']
types = types.concat types.map {|t| t + '*'}

1.upto(n) do |i|
  desc_exp(types, i).map do |f|
    fpname = gen_funcptr_name(f)
    params = f.join(',')
    lines0.push gen_funcptr_def('void', fpname, params)
    args = gen_args(f)
    lines1.push gen_execute_kernel(fpname, args)
  end
end

puts '#ifndef _AUTOGEN_H_'
puts '#define _AUTOGEN_H_'
puts '#include "railgun.h"'
puts '#include <string.h>'
puts '#include <cuda_runtime.h>'

lines0.each {|s| puts(s)}

puts 'void _execute_kernel(const char* fmt, railgun_task* task, railgun_memory* mem, cudaStream_t *strm) {'
puts lines1.join("\nelse ")
puts '}'
puts '#endif'
