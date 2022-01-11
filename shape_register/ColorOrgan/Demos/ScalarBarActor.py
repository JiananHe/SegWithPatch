import vtk


def main():
    # Create a sphere for some geometry
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(0, 0, 0)
    sphere.SetRadius(1)
    sphere.Update()

    # Create scalar data to associate with the vertices of the sphere
    numPts = sphere.GetOutput().GetPoints().GetNumberOfPoints()
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfValues(numPts)
    for i in range(numPts):
        scalars.SetValue(i, float(i)/numPts)
    poly = vtk.vtkPolyData()
    poly.DeepCopy(sphere.GetOutput())
    poly.GetPointData().SetScalars(scalars)

    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    array = poly.GetPointData().GetScalars()
    numpy_nodes = vtk_to_numpy(array)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToMapScalars()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    scalarBarActor = vtk.vtkScalarBarActor()
    scalarBarActor.SetLookupTable(mapper.GetLookupTable())
    scalarBarActor.SetTitle("Title")
    scalarBarActor.SetNumberOfLabels(4)

    # Create a lookup table to share between the mapper and the scalarbar
    hueLut = vtk.vtkLookupTable()
    hueLut.SetTableRange(0, 1)
    hueLut.SetHueRange(0.5, 1)
    hueLut.SetSaturationRange(0.5, 1)
    hueLut.SetValueRange(1, 1)
    hueLut.Build()

    mapper.SetLookupTable(hueLut)
    scalarBarActor.SetLookupTable(hueLut)

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    renderer.GradientBackgroundOn()
    renderer.SetBackground(1, 1, 1)
    renderer.SetBackground2(0, 0, 0)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Create an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actors to the scene
    renderer.AddActor(actor)
    renderer.AddActor2D(scalarBarActor)

    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == "__main__":
    main()
