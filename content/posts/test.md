---
title: "TorchScript的转化"
subtitle:
date: 2023-12-13T12:18:21+08:00
draft: false
author:
  name:
  link:
  email:
  avatar:
description: 
keywords: 
license:
comment: false
weight: 0
tags:
  - jit
categories:
  - torch
hiddenFromHomePage: false
hiddenFromSearch: false
hiddenFromRss: false
summary:
resources:
  - name: featured-image
    src: featured-image.jpg
  - name: featured-image-preview
    src: featured-image-preview.jpg
toc: true
math: false
lightgallery: false
password:
message:
repost:
  enable: true
  url:

# See details front mat`te`r: https://fixit.lruihao.cn/documentation/content-management/introduction/#front-matter
---


[TOC]


<p class=para>
本文将以一个 pytorch demo model 为例，展示 pytorch 是如何实现模型的 script 化的，即如何将 nn.module 转成 torch.jit.RecursiveScriptModule 的过程中，pytorch 底层到底做了哪些工作。
</p>

```Python TI:"run.py" HL:"12"
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        a = torch.tensor([10, 10, 1], dtype=torch.float32)
		out = a + x
        return out
model = Model()
script_model = torch.jit.script(model)
input_ = torch.ones([10, 10, 1], dtype=torch.float32)
output = vulkan_script(input_)
print(output)
```

^106e4a

<div class=para> 
上面这段 demo 代码: 
</div>
1. 定义了一个最简单的模型，没有任何多余的属性和变量，只是创建了一个 <font color=red>Tensor a</font> ，将其与输入的<font color=red>Tensor x</font>进行加运算得到输出。
2. 创建了一个模型的实例 model。
3. 通过 torch. jit. script 方法将 model 从进行 script 化，得到模型 script_model。
4. 创建输入变量并送入 script_model 进行计算，得到输出。

<div class=para> 
torch.jit.script 方法模型正是对模型进行 script 化的方法之一，那么其到底做了哪些工作呢？跟随 pdb 和 gdb 一起追踪一下整个过程。 
</div>

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_script.py:script" HL:"11-12"
def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
	...
    if isinstance(obj, torch.nn.Module):
        obj = call_prepare_scriptable_func(obj)
        return torch.jit._recursive.create_script_module(
				            obj, torch.jit._recursive.infer_methods_to_compile
        )
```

<p class=para> 
script 方法主要接受的参数是需要 script 化的对象 obj，这个对象可以是很多中类型，如一个方法，一个列表，一个字典，一个 nn.Module 等.
script 方法针对这些类型都有 script 化的方法。其他参数一般不传。在删减掉不重要的代码后，核心的操作是 jit._recursive. create_script_module 方法的调用，其参数为需要 script 化的对象和一个名为 torch.jit._recursive.infer_methods_to_compile 的方法。
</p>

# 1.infer_methods_to_compile 方法

<p class=para>
那么这个 infer_methods_to_compile 做了些什么？
先注释，后代码
</p>

```text
    """
    Implements the default rules for which methods should act as starting
    points for compilation (TODO add a link when the rules are published).
    """
```

<p class=para>
好像没太看懂，看看代码
</p>

```python TI: "/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:infer_methods_to_compile" HL:"13-19"
def infer_methods_to_compile(nn_module):
    check_module_initialized(nn_module)
    user_annotated_ignored_attributes = getattr(
        nn_module, "__jit_ignored_attributes__", list()
    )
    ignored_properties = jit_ignored_properties(nn_module)
    methods: List[str] = []
    if hasattr(nn_module, "forward") and not _jit_internal.is_ignored_fn(
        nn_module.forward
    ):
        forward_func = getattr(nn_module.forward, "__func__", None)
        module_forward = getattr(torch.nn.Module, "forward", None)
        if forward_func != module_forward:
            methods = ["forward"]
    exported = []
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.EXPORT
        ):
            exported.append(name)
    methods = methods + exported
    overload_name_mappings = dict(getattr(nn_module, "__overloads__", {}))
    overload_info = get_overload_annotations(nn_module, ignored_properties)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)
    nn_module.__overloads__ = overload_name_mappings
    # we shouldn't directly compile overloaded methods, just its overloads
    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings
    filtered_methods = filter(ignore_overloaded, methods)
    # Unique the methods. We don't want to use a set to store the methods because it
    # introduces non-determinism to compile order.
    uniquer: Set[str] = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)
    stubs = []
    for method in uniqued_methods:
        stubs.append(make_stub_from_method(nn_module, method))
    return overload_stubs + stubs
```

<p class=para>
infer_methods_to_compile 方法只接受一个参数，就是 nn. Module 类型的对象，并且一定是要调用过父类的初始化方法的，这个在方法内部会做检查，如果不符合直接抛出异常。
简单来说，infer_methods_to_compile 就是结合各种条件，从 nn. Module 对象的中选出需要进行编译的所有方法（尤其是 forward 方法），然后调用 make_stub_from_method 函数对这些方法进行处理，最后把生成的结果返回。
</p>

## 1.1 make_stub_from_method 方法
<p class=para>
那么，make_stub_from_method 又做了哪些事情呢？
</p>

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:make_stub_from_method" HL:"13"
def make_stub_from_method(nn_module, method_name):
    func = getattr(nn_module, method_name)
    if isinstance(func, ScriptMethodStub):
        return func
    # Make sure the name present in the resulting AST will match the name
    # requested here. The only time they don't match is if you do something
    # like:
    #   def _forward(self):
    #       pass
    #   forward = _forward
    # In this case, the actual function object will have the name `_forward`,
    # even though we requested a stub for `forward`.
    return make_stub(func, method_name)
```
```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:make_stub"
def make_stub(func, name):
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = get_jit_def(func, name, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)
```
^e01a53
<p class=para>
可以看到，make_stub_from_method 实际上是从 nn. module 中拿到了 method_name 对应的方法，然后调用了 make_stub 函数
</p>

### 1.1.1 make_stub
<p class=para>
make_stub 函数接受两个参数，一个是方法本身，还有就是对应的方法名
从返回的值来看，返回的类型是一个 namedtuple，名为 ScriptMethodStub，其定义如下：
</p>

```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:ScriptMethodStub"
ScriptMethodStub = collections.namedtuple(
    "ScriptMethodStub", ("resolution_callback", "def_", "original_method")
)
```
<p class=para>
named_tuple 类似于 Dict 和 Tuple 的集合，其对所包含值的索引即可以通过 index 的方式也可以通过 key 的方式
这个名为 ScriptMethodStub 的 namedtuple 共包含三个成员，分别为：
</p>

- resolution_callback 
- def_
- original_method

<p class=para>
original_method 很好理解，就是函数 func 本身，那剩下的 resolution_callback 和 def_分别又是什么呢？这需要从返回他们的方法来查找
</p>

#### 1.1.1.1 createResolutionCallbackFromClosure

首先，从 [[#^e01a53|make_stub函数]] 可以看到，关于 resolution_callback 的获取是通过 createResolutionCallbackFromClosure 方法，代码逻辑比较复杂，但有一段注解值得一看：

```text TI="/home/shanlin/disk/debug/pytorch/torch/_jit_internal.py:createResolutionCallbackFromClosure"
# [local resolution in python]
# Depending on where a variable is defined, and where it is used, we may
# or may not be able to recover its value when recursively compiling a
# script function. Remember in the general case, a module or function is
# first defined and then later scripted. This means we do not have a
# chance to capture the active frames when the function is defined. Hence any
# name resolution has to happen later on the created closure. The way
# python captures type annotations restricts what we can recover. The
# follow example illustrates the different cases:
#
#         class MyGlobalClass:
#         ...
#         def my_local_scope():
#             @torch.jit.script
#             class MyClass:
#                 ...
#             @torch.jit.script
#             class MyClassUsedAsVar:
#                 ...
#             def eg(x: MyClass, y: MyGlobalClass):
#                 a_local_capture : Foo
#                 return MyClassUsedAsVar(x)
#
# MyGlobalClass is defined in the __globals__ dictionary of function
# 'eg', so it is always recoverable. my_local_scope introduces a new local
# variable scope in the function. Classes defined here are only visible as
# local variables. For the case of MyClassUsedAsVar, it is captured
# because it is used as a variable inside the body of the function, and we
# can resolve it using the captures returned from `get_closure`. However,
# the type annotations are not captured by the closure. In Python
# 3.0--3.9, the _value_ of MyClass and MyGlobalClass will be available as
# annotations on `eg``, but starting in Python 4.0, they will represented as
# strings and no longer present. Furthermore, since the body of `eg` does
# not reference those names, they do not appear in the list of closed over
# variables. In Python 2.x, type annotations are in comments, leading to a
# similar situation where their definitions are not available. We anticipate
# that most users will not run into this issue because their modules and
# functions will be defined at a global scope like MyGlobalClass. In cases
# where they are not, it is possible to work around issues by declaring the
# values global in the function.
# In Python 3.9 declaring class as global will make it invisible to
# `inspect.getsource`, see https://bugs.python.org/issue42666 .
# This could be worked around by manualy adding it to `global()` dictionary.
```
>"Local resolution"在 Python 中指的是名称解析的本地规则。
当我们引用一个变量/函数/类名称时,Python 会按照一定顺序查找该名称对应的对象:

>1. 本地作用域:首先在当前函数/方法等作用域中查找该名称。
>2. 内置作用域(Built-in Scope):如果在本地未找到,则在Python内置名称空间如builtins中查找。
>3. 全局作用域:如果前两步都没有找到,则在当前模块的全局作用域中查找。
>4. 嵌套作用域(Enclosing Scope):之后会依次向外层函数/类的作用域查找。
>5. 第三方模块作用域:最后搜索当前导入的所有第三方模块。<br/>
>
>所以我们把第一步称为"本地解析"(Local Resolution)。这意味着,当解析一个名称时,Python 首先会在当前最小的本地作用域里寻找匹配的对象。

<p class=para>
理解了上面的内容之后，总结下来就是由于 pytorch 的 module 或方法总是先被定义再被 script 化，这导致 torch. jit 模块没有机会捕捉到激活的栈帧，所以在后来所创建的闭包上进行变量的解析。
</p>
<p class=para>
所以这个 resolution_callback 函数是一种回调函数，用于从闭包中解析变量。
</p>

<font color=yellow>那么 def_又是什么呢？</font>
#### 1.1.1.2 get_jit_def

<p class=para>
先注释，后代码
</p>

```text
    """
    Build a JIT AST (TreeView) from the given function.

    Args:
        fn: A function object to compile or a pre-parsed ParsedDef object
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: **If** this function is a method, what the type name of `self` is.
    """
```

<p class=para>
介绍很简单，一句话交代了这个函数的作用：从给定的函数建立一棵抽象语法树。再看代码：
</p>

```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/frontend.py:get_jit_def"
def get_jit_def(fn, def_name, self_name=None, is_classmethod=False):

    parsed_def = parse_def(fn) if not isinstance(fn, _ParsedDef) else fn
    type_line = torch.jit.annotations.get_type_line(parsed_def.source)
    fn_def = parsed_def.ast.body[0]

    if is_classmethod:
        arg_name = fn_def.args.args[0].arg
        # Insert a statement that assigns the first argument to the class
        assign_stmt = ast.parse(f"{arg_name} = {self_name}").body[0]
        fn_def.body.insert(0, assign_stmt)

    # Swap out the function signature and body if it is unused
    if should_drop(fn):
        unused_fn_def = ast.parse(
            'def unused_fn(self: Any):\n\traise RuntimeError("Cannot call @unused methods")'
        )
        if len(unused_fn_def.body) != 1 or not isinstance(
            unused_fn_def.body[0], ast.FunctionDef
        ):
            raise RuntimeError(
                f"Expected a single top-level function: {parsed_def.filename}:{parsed_def.file_lineno}"
            )
        unused_def = unused_fn_def.body[0]
        fn_def.body = unused_def.body
        # kwarg/vararg not supported by `build_def`
        fn_def.args.kwarg = fn_def.args.vararg = None
        for arg in fn_def.args.args + fn_def.args.kwonlyargs:
            # Replace potentially unsupported type annotations by "Any"
            arg.annotation = unused_def.args.args[0].annotation
        if _is_drop_fn(fn):
            # Dropping potentially unsupported return type annotation for jit._drop
            fn_def.returns = None
            fn_def.type_comment = None

    # If MonkeyType is installed, get all the consolidated type traces
    # for the arguments from type_trace_db
    type_trace_db = torch.jit._script._get_type_trace_db()
    pdt_arg_types = None
    if monkeytype_trace and not isinstance(fn, _ParsedDef):
        qualname = get_qualified_name(fn)
        pdt_arg_types = type_trace_db.get_args_types(qualname)

    return build_def(
        parsed_def.ctx,
        fn_def,
        type_line,
        def_name,
        self_name=self_name,
        pdt_arg_types=pdt_arg_types,
    )
```

<p class=para>
大体来看，这里是调用了第三方库 ast 解析函数 fn 生成了一颗生成树，并根据这颗生成树，以及相关信息，调用 build_def 构建了一个数据结构 Def 的数据结构，build_def 函数的具体实现如下：
</p>

```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/frontend.py:build_def"
def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None):
    body = py_def.body
    r = ctx.make_range(py_def.lineno, py_def.col_offset, py_def.col_offset + len("def"))

    param_list = build_param_list(ctx, py_def.args, self_name, pdt_arg_types)
    return_type = None
    if getattr(py_def, "returns", None) is not None:
        return_type = build_expr(ctx, py_def.returns)

    decl = Decl(r, param_list, return_type)
    is_method = self_name is not None
    if type_line is not None:
        type_comment_decl = torch._C.parse_type_comment(type_line)
        decl = torch._C.merge_type_from_type_comment(decl, type_comment_decl, is_method)

    return Def(Ident(r, def_name), decl, build_stmts(ctx, body))
```

<p class=para>
其实这些数据结构，包括 Def、Decl、Ident 等都是 torch. jit 自己定义的抽象语法树的结点的一种，详情请见 [[3.toch.jit中的抽象语法树|torch.jit中的抽象语法树]]，本文不做详述。
</p>
<p class=para>
其中，Def 可以理解为以一个函数为基础，构造出的抽象语法树的根节点。
</p>

#### 1.1.1.3 backtrace

<p class=para>
让我们回到梦开始的地方：
</p>

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:make_stub"
def make_stub(func, name):
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = get_jit_def(func, name, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)
```

<p class=para>
现在我们明白了，在 ScriptMethodStub 这个 namedtuple 中包含三个成员，分别是用于解析函数中变量的回调函数 rcb 和函数的抽象生成树 ast，以及原始的函数 func，他们一起组成了 ScriptMethodStub。
</p>
<p class=para>
那么接着向上回溯：
</p>

```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:infer_methods_to_compile"
	......

    for method in uniqued_methods:
        stubs.append(make_stub_from_method(nn_module, method))
    return overload_stubs + stubs
```

<p class=para>
所以现在可以得出结论，infer_methods_to_compile 这个方法的作用就是从 nn. module 中挑选出需要编译的方法，并对他们分别进行处理，生成相应的解析回调函数和抽象语法树，封装成数据结构 ScriptMethodStub 中返回。
</p>

---


# 2. torch.jit.\_recursive.create_script_module

<p class=para>
那么，create_script_module 方法中，具体又做了什么事情呢？
</p>
<p class=para>
先注释，后代码
</p>

```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:create_script_module"
    """
    Creates a new ScriptModule from an nn.Module

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
        share_types:  Whether to share underlying JIT types between modules (if possible).
            NOTE: Only set to False this when we cannot guarantee type sharing will work
                correctly. This only happens today for traced modules, where the same
                module can produce different traced methods depending on the inputs.
        is_tracing: Whether this function is called during tracing or scripting. If tracing,
                we don't need to do AttributeTypeIsSupportedChecker because all the unsupported
                attributes will be baked as constant in the tracing graph. In addition,
                this check significantly slows down the traced modules when the module size is big.
    """
```

<p class=para>
言简意赅，从 nn. Module 创建一个新的 ScriptModule，看来真正干活的就是这个函数了
</p>
<p class=para>
create_script_module 方法接受四个参数，前两个我们很熟悉，分别是要转换的 nn_module，一个用于产生 ScriptMethodStubs 的方法 （infer_methods_to_compile）。后面两个参数 share_types 和 is_tracing 分别又是什么意思呢？
</p>
<p class=para>
从注释上看，share_types 用于只是是否在多个 module 之间共享底层的 JIT 类型。
</p>
<p class=para>
而 is_tracing 则用于表示 create_script_module 函数是否是在 trace 的过程中被调用的，如果是在 trace 的过程中调用的，那么就不会再调用 AttributeTypeIsSupportedChecker 这个方法，因为再 trace 的过程中，所有不支持的属性都会再 trace graph 中被备份为常量，且 AttributeTypeIsSupportedChecker 这个函数会很大程度的降低执行效率，所以通过传参的方式来确定当前状态，从而避免不必要的性能开销。
</p>
<p class=para>
同时，我们也能逆向推出，AttributeTypeIsSupportedChecker 这个方法适用于检测 nn_module 中定义的哪些类型和语法是不被 jit scirpt 所支持的。
</p>
<p class=para>
create_script_module 中并没有太多事情，主要包括：
</p>

- 检查 module 是否已经被初始化
- 获取 module 的 concrete_type
- 调用 create_script_module_impl 方法


关于 concrete_type 的内容可以参考 [[4.concrete_type相关内容|concrete_type相关内容]]

<p class=para>
那么看来，实际上真正 work 的是⭐ create_script_module_impl 这个函数 
</p>

## 2.1 create_script_module_impl

<p class=para>
从注释来看，create_script_module_impl 函数完成了一个 pytorch 模型从 nn. Module 到 torch. jit. RecursiveScriptModule 的转换。
</p>

```text
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
```
<p class=para>
共接受三个参数，分别是待转换的 nn. Module 及其对应的 concrete type，最后是一个用于产生 ScriptMethodStub 的函数。
</p>
<p class=para>
create_script_module_impl 的代码有点长，主要拆开四个部分来分析：
</p>

### 2.1.1 构造前的准备工作

<p class=para>
RecursiveScriptModule 构造前需要做哪些准备工作呢？
</p>

<p class=para>
主要分为 3点，1）核心成员的构建，2）stubs 的构建以及 3）初始化函数的定义
</p>

查看 [[torch.jit.RecursiveScriptModule#^b6d92f|RecursiveScriptModule 的构造函数]]可以看到，RecursiveScriptModule 的核心类其实是一个 c++编写的 [[torch.jit.Module|torch. jit.Module]] 类，而准备工作的内容之一，就是构造这个 Module 类。

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:create_script_module_impl"
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)
    method_stubs = stubs_fn(nn_module)
    property_stubs = get_property_stubs(nn_module)
    hook_stubs, pre_hook_stubs = get_hook_stubs(nn_module)
    
    ......
```

<p class=para>
准备工作的另一项工作则是为 nn. module 的 property，method 以及 hook 和 pre_hook 等生成 stub。
</p>

<p class=para>
准备工作的最后一项，就是为 RecursiveScriptModule 定义一个初始化函数，这个函数用于为新构建的 RecursiveScriptModule 实例进行初始化工作，包括属性、参数、缓冲（buffer）的拷贝，子模块的拷贝以及一些被修饰器@unused/@ignore 修饰的方法的拷贝
</p>

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:create_script_module_impl"
    def init_fn(script_module):
        # Initialize the ScriptModule:
        # 1. Copy the attributes/parameters/buffers from the original `nn_module` to the new ScriptModule.
        for name in concrete_type.get_attributes().keys():
            orig_value = getattr(nn_module, name)
            orig_value = (
                orig_value.value
                if isinstance(orig_value, torch.jit.Attribute)
                else orig_value
            )
            cpp_module.setattr(name, orig_value)

        # 2. Copy the submodules from the original `nn_module` to the new ScriptModule,
        #    recursively scripting them.
        for name, sub_concrete_type in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            assert isinstance(
                orig_value, Module
            ), f"Expected Module but got {type(orig_value)}"
            module_type = sub_concrete_type.jit_type
            if isinstance(module_type, torch._C.InterfaceType):
                # use the interface inference rule to compile the module
                scripted = interface_script(module_type, orig_value)
            elif isinstance(orig_value, torch.jit.ScriptModule):
                scripted = orig_value
            else:
                # always reuse the provided stubs_fn to infer the methods to compile
                scripted = create_script_module_impl(
                    orig_value, sub_concrete_type, stubs_fn
                )

            cpp_module.setattr(name, scripted)
            script_module._modules[name] = scripted

        # 3. Copy @ignored/@unused methods and attrs from the original `nn_module` to the new ScriptModule.
        #    This ensures we can access these Python methods on the ScriptModule.
        for name in dir(nn_module):
            if name in ignored_properties:
                continue
            item = getattr(nn_module, name, None)
            if inspect.ismethod(item) and _jit_internal.is_ignored_fn(item):
                unbound_function = getattr(nn_module, name).__func__
                bound_method = unbound_function.__get__(script_module)
                setattr(script_module, name, bound_method)
            elif concrete_type.is_ignored_attribute(name):
                setattr(script_module, name, item)

        # For convenience, attach the concrete type to the new ScriptModule
        script_module._concrete_type = concrete_type

```

### 2.1.2 RecursiveScriptModule 的构造
 参考 [[torch.jit.RecursiveScriptModule#^b180f8|RecursiveScriptModule 类构建函数]]，

```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:create_script_module_impl"
......

    # Actually create the ScriptModule, initializing it with the function we just defined
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

......

```



### 2.1.3 类方法的编译
<p class=para>
如果之前没有为当前的 concrete_type 编译过相应的方法，那么需要对其进行编译
</p>
<p class=para>
主要包括方法、属性的 get、set 方法以及 hook 和 pre_hook 等，调用的函数是 create_methods_and_properties_from_stubs
</p>

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:create_methods_and_properties_from_stubs"
def create_methods_and_properties_from_stubs(
    concrete_type, method_stubs, property_stubs
):
    method_defs = [m.def_ for m in method_stubs]
    method_rcbs = [m.resolution_callback for m in method_stubs]
    method_defaults = [get_default_args(m.original_method) for m in method_stubs]

    property_defs = [p.def_ for p in property_stubs]
    property_rcbs = [p.resolution_callback for p in property_stubs]

    concrete_type._create_methods_and_properties(
        property_defs, property_rcbs, method_defs, method_rcbs, method_defaults
    )

```

<p class=para>
可以看到，create_methods_and_properties_from_stubs 是通过 c++函数_create_methods_and_properties 来为 c++ Module 类创建方法和属性的，
</p>
这个过程 GraphFunction 的构建，其中又包括静态计算图的构建以及优化等，具体的内容参考 [[2. GraphFunction的构建|GraphFunction的构建]]
### c++ 接口的暴露
<p class=para>
当所有工作都结束了之后，那么需要将 c++的接口暴露出来，让用户能够在 python 端调用，所以下面的代码主要是为了暴露 c++ Module 类的方法、属性等
</p>
```python TI="/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:create_script_module_impl"
......
    # Make the compiled methods available to the Python ScriptModule class.
    for method_stub in method_stubs:

        name = method_stub.original_method.__name__
        script_method = cpp_module._get_method(name)
        # Wrap the original to propagate docstrings and such.
        # TODO: we don't currently do this functions that are recursively
        # compiled, we should.
        wrapped_script_method = functools.wraps(method_stub.original_method)(
            script_method
        )

        # Add the methods to the script_module directly. This ensures they will
        # be found first when `name` is looked up (as opposed to the stubs or
        # nn.Module.forward)
        script_module.__dict__[name] = wrapped_script_method

    # Make module properties available on the Python ScriptModule class.
    for property_stub in property_stubs:
        property_name = property_stub.def_.name().name
        fget = cpp_module._get_method(property_stub.def_.getter_name().name)
        # Setter is optional, so it may not exist.
        setter_name = property_stub.def_.setter_name()
        fset = cpp_module._get_method(setter_name.name) if setter_name else None
        script_module.__dict__[property_name] = property(property_name, fget, fset)  # type: ignore[arg-type]

    # copy over python methods to script module if they aren't defined on the script module
    # this is currently an internal api used only on module containers
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if (
            _jit_internal.get_torchscript_modifier(item)
            is _jit_internal.FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
        ):
            add_python_attr_to_scripted_model(script_module, nn_module, name)

    return script_module


```

```python TI:"/home/shanlin/disk/debug/pytorch/torch/jit/_recursive.py:add_python_attr_to_scripted_model"
def add_python_attr_to_scripted_model(script_model, orig, attr):
    if hasattr(orig, attr) and script_model_defines_attr(script_model, attr):
        setattr(script_model, attr, getattr(orig, attr))
```
<!--more-->
