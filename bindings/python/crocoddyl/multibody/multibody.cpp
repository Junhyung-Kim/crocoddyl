///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeFrames();
  exposeFrictionCone();
  exposeWrenchCone();
  exposeCoPSupport();
  exposeStateMultibody();
  exposeStateKinodynamic();
  exposeActuationFloatingBase();
  exposeActuationKinoBase();
  exposeActuationFull();
  exposeActuationModelMultiCopterBase();
  exposeForceAbstract();
  exposeContactAbstract();
  exposeImpulseAbstract();
  exposeContactMultiple();
  exposeImpulseMultiple();
  exposeDataCollectorMultibody();
  exposeDataCollectorContacts();
  exposeDataCollectorImpulses();
  exposeDifferentialActionFreeFwdDynamics();
  exposeDifferentialActionKinoDynamics();
  exposeDifferentialActionContactFwdDynamics();
  exposeActionImpulseFwdDynamics();
  exposeResidualState();
  exposeResidualFlyState();
  exposeResidualFlyState1();
  exposeResidualCentroidalMomentum();
  exposeResidualCentroidalAngularMomentum();
  exposeResidualCoMPosition();
  exposeResidualCoMKinoPosition();
  exposeResidualContactForce();
  exposeResidualContactFrictionCone();
  exposeResidualContactCoPPosition();
  exposeResidualContactWrenchCone();
  exposeResidualContactControlGrav();
  exposeResidualControlGrav();
  exposeResidualFramePlacement();
  exposeResidualKinoFramePlacement();
  exposeResidualFrameRotation();
  exposeResidualKinoFrameRotation();
  exposeResidualFrameTranslation();
  exposeResidualFrameVelocity();
  exposeResidualImpulseCoM();

#ifdef PINOCCHIO_WITH_HPP_FCL
  exposeResidualPairCollision();
#endif

  exposeCostState();
  exposeCostCoMPosition();
  exposeCostControlGrav();
  exposeCostControlGravContact();
  exposeCostCentroidalMomentum();
  exposeCostFramePlacement();
  exposeCostFrameTranslation();
  exposeCostFrameRotation();
  exposeCostFrameVelocity();
  exposeCostContactForce();
  exposeCostContactWrenchCone();
  exposeCostContactImpulse();
  exposeCostContactCoPPosition();
  exposeCostContactFrictionCone();
  exposeCostImpulseCoM();
  exposeCostImpulseFrictionCone();
  exposeCostImpulseWrenchCone();
  exposeCostImpulseCoPPosition();
  exposeContact1D();
  exposeContact2D();
  exposeContact3D();
  exposeContact6D();
  exposeImpulse3D();
  exposeImpulse6D();
}

}  // namespace python
}  // namespace crocoddyl
