import datetime
import logging


class Pipeline:
    def __init__(self, source, processors, sinks=None, handle_exceptions=False):
        self.handle_exceptions = handle_exceptions
        self.source = source
        self.processors = list(processors)
        self.sinks = sinks if sinks else []
        self.started = False

    def __start(self):

        def call_start(obj):
            if hasattr(obj, 'start'):
                obj.start()

        self.started = True
        call_start(self.source)
        for p in self.processors:
            call_start(p)
        for s in self.sinks:
            call_start(s)

    def stop(self):
        def call_stop(obj):
            if hasattr(obj, 'stop'):
                obj.stop()

        self.started = False
        call_stop(self.source)
        for p in self.processors:
            call_stop(p)
        for s in self.sinks:
            call_stop(s)

    def run(self):
        if not self.started:
            self.__start()
        try:
            data = self.source.get_data()
        except Exception as exc:
            logging.exception(exc)
            if not self.handle_exceptions:
                raise
            return
        for processor in self.processors:
            try:
                data = processor.process(data)
            except Exception as exc:
                logging.exception(f"Running processor {type(processor)}")
                if not self.handle_exceptions:
                    raise
        for sink in self.sinks:
            try:
                sink.put_data(data)
            except Exception as exc:
                logging.exception(f"Running sink {type(sink)}")
                if not self.handle_exceptions:
                    raise
        if data.get('stop_pipeline', False):
            return False
        return True

    def run_k(self, k):
        for _ in range(k):
            self.run()
        if self.started:
            self.stop()
            self.started = False

    def run_while_source(self):
        result = True
        while result:
            result = self.run()
        if self.started:
            self.stop()
            self.started = False


class CustomSink:
    def __init__(self, callback):
        self.callback = callback

    def put_data(self, data):
        self.callback(data)


class CustomProcessor:
    def __init__(self, callback):
        self.callback = callback

    def process(self, data):
        data = self.callback(data)
        return data


class CustomSource:
    def __init__(self, callback):
        self.callback = callback

    def get_data(self):
        return self.callback()
