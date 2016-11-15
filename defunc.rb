#!/usr/bin/env ruby

# Descartes Exponentiation
def desc_exp(arr, n)
  if n > 1
    arr.product(desc_exp(arr, n - 1)).map {|x| x.flatten}
  else
    arr.map {|x| [x]}
  end
end

# generate definition of function pointer for C language
def gen_funcptr_def(type, fmt)
  fpname = fmt.map {|s| s[-1] == '*' ? s[0] : s[0].upcase}.join + '_f'
  args = fmt.join(',')
  "typedef #{type} (*#{fpname})(#{args});"
end

# generate definition of function

n = ARGV[0].to_i
types = ['char', 'int', 'float', 'double']
types = types.concat types.map {|t| t + '*'}

1.upto(n) do |i|
  desc_exp(types, i).map do |f|
     puts gen_funcptr_def('void', f)
  end
end
