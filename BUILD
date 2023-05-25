load(":build_defz.bzl", "hk_py_binary", "hk_py_library")

hk_py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":cifar",
        ":loss",
        ":models",
        # pip: absl:app
        # pip: bsuite/environments:catch
        # pip: jax
        # pip: optax
    ],
)

hk_py_library(
    name = "loss",
    srcs = ["loss.py"],
    deps = [
        ":models",
        # pip: dm_env
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        # pip: dm_env
        # pip: jax
        # pip: numpy
    ],
)

hk_py_library(
    name = "cifar",
    srcs = ["cifar.py"],
    deps = [
        # pip: dm_env
        # pip: jax
        # pip: numpy
    ],
)
