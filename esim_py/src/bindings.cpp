#include <esim.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

PYBIND11_MODULE(esim_py, m) {
    m.doc() = "ESIM bindings";

    py::class_<EventSimulator>(m, "EventSimulator")
        .def(py::init<float,float,float,float,bool>())
        .def("generateFromFolder", &EventSimulator::generateFromFolder, py::return_value_policy::reference_internal)
        .def("generateFromVideo", &EventSimulator::generateFromVideo, py::return_value_policy::reference_internal)
        .def("generateFromStampedImageSequence", &EventSimulator::generateFromStampedImageSequence, py::return_value_policy::reference_internal)
        .def("setParameters", &EventSimulator::setParameters);
}