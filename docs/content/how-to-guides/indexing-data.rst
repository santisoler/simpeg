.. _indexing_data:

Indexing data
=============

Survey objects (like the
:class:`simpeg.electromagnetics.time_domain.survey.Survey`) contain collections
of sources (e.g.
:class:`simpeg.electromagnetics.time_domain.sources.CircularLoop`) that are
linked to a set of receiver objects (e.g.
:class:`simpeg.electromagnetics.time_domain.receivers.PointMagneticFluxDensity`).
Each one of the receiver objects can have multiple receiver locations.

We usually use survey objects like this one inside simulations.
 (e.g.
:class:`simpeg.electromagnetics.time_domain.Simulation1DLayered`).
The predicted data that the simulation can return will always be a 1D Numpy
array, containing the predicted data values on each one of the receiver
locations for each one of their corresponding sources.

Sometimes we want to access only the data predicted at a particular receiver,
for a particular source. We can use the
:meth:`~simpeg.survey.BaseSurvey.get_slice()` method of the survey object to
get the elements of the data array that belong to that combination of source
and receiver.

For example, let's consider a survey in which we have a couple of
:class:`~simpeg.electromagnetics.time_domain.receivers.PointMagneticFluxDensity`
receivers, and a single
:class:`~simpeg.electromagnetics.time_domain.sources.CircularLoop` that we move
between two different locations.

.. ipython::

   import numpy as np

   times = np.logspace(-5, -2, 31)  # time channels (s)
   receivers_location = np.array(  # define three receivers along x
      [
         [-5.0, 0.0, 20.0],
         [0.0, 0.0, 20.0]
         [5.0, 0.0, 20.0]
      ]
   )
   receiver = tdem.receivers.PointMagneticFluxDensity(
     receiver_location, times, orientation=receiver_orientation
   )
