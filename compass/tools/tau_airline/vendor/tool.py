"""Minimal stand-in for tau_bench.envs.tool.Tool (vendored tools subclass it)."""


class Tool:
    @staticmethod
    def invoke(data, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info():
        raise NotImplementedError
